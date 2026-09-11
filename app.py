import sys
import io
import traceback
import json
import re
import subprocess
import tempfile
import os

from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.colab import userdata


# ============================================================
# 1. GEMINI SETUP
# ============================================================

try:
    api_key = userdata.get('GEMINI_API_KEY')

    genai.configure(api_key=api_key)

    print("API Key configured successfully.")

except userdata.SecretNotFoundError:
    print("Error: GEMINI_API_KEY not found in Colab Secrets")
    api_key = None


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ============================================================
# 2. SHARED LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    retry_count: int
    test_cases: Optional[str]


# ============================================================
# 3. PYTHON CODE EXECUTION TOOL
# ============================================================

@tool
def run_python_code(code: str, test_input: str = "") -> str:
    """Execute Python code with isolated input and return clean output."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    temp_file_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as temp_file:

            temp_file.write(clean_code)
            temp_file_path = temp_file.name

        result = subprocess.run(
            [sys.executable, temp_file_path],
            input=test_input,
            text=True,
            capture_output=True,
            timeout=10
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode != 0:

            if output and error:
                return output + "\n" + error

            if error:
                return error

            return output if output else "Execution Error"

        if output:

            lines = output.splitlines()

            cleaned_lines = []

            for line in lines:

                if line.startswith("Enter ") and ":" in line:
                    line = line.split(":", 1)[1].strip()

                if line:
                    cleaned_lines.append(line)

            output = "\n".join(cleaned_lines).strip()

        return output

    except subprocess.TimeoutExpired:

        return "Execution Error: Program exceeded 10 seconds."

    except Exception:

        return f"Execution Error:\n{traceback.format_exc()}"

    finally:

        if temp_file_path:

            try:
                os.remove(temp_file_path)
            except:
                pass


# ============================================================
# 4. TEST CASE GENERATOR TOOL
# ============================================================

@tool
def generate_test_cases(task_description: str, code: str) -> str:
    """Generate independent executable test cases from the task specification."""

    prompt = f"""
You are an independent Senior QA Engineer.

Your job is to create test cases for a Python coding task.

IMPORTANT:
You must determine the expected output from the ORIGINAL TASK.
Do NOT determine the expected output by simply following the generated code.

ORIGINAL TASK:
{task_description}

GENERATED CODE:
{code}

Your responsibilities:

1. Understand the requirements in the original task.
2. Determine the correct behavior independently.
3. Generate exactly 3 executable test cases.
4. Inputs must match the input format required by the task.
5. Expected outputs must be calculated from the task requirements.
6. Use different meaningful cases:
   - normal case
   - boundary case
   - edge case
7. Do not copy incorrect behavior from the generated code.
8. Do not assume behavior that is not specified by the task.
9. If an edge case is undefined by the task, choose a reasonable standard interpretation.
10. Expected output must contain ONLY the program's required output.
11. Do not include input prompts in expected output.
12. For genuinely empty expected output, use "".
13. Test cases must be independent.
14. Inputs must be dynamically generated according to the task.

Examples:

Task:
Write a Python program to find the factorial of a number.

Good tests:
5 -> 120
0 -> 1
1 -> 1

Task:
Write a Python program to check whether a number is prime.

Good tests:
7 -> Prime
1 -> Not Prime
2 -> Prime

Task:
Write a Python program to find the largest number in a list.

Good tests:
3
10 20 5 -> 20

Return ONLY valid JSON.

Required format:

[
  {{
    "input": "input values separated by newline",
    "expected": "correct expected output"
  }},
  {{
    "input": "input values separated by newline",
    "expected": "correct expected output"
  }},
  {{
    "input": "input values separated by newline",
    "expected": "correct expected output"
  }}
]

Do not include markdown.
Do not include explanations.
"""

    response = llm_flash.invoke(prompt)

    content = (
        response.content
        if hasattr(response, "content")
        else str(response)
    )

    if isinstance(content, list):

        content = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    content = str(content).strip()

    content = (
        content
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return content


# ============================================================
# 5. TASK INPUT AGENT
# ============================================================

def task_input_node(state: CrewState):

    print("\n")
    print("=" * 60)
    print("              TASK INPUT AGENT")
    print("=" * 60)

    task = input(
        "\nEnter coding task (or type 'exit'): "
    ).strip()

    if task.lower() == "exit":

        return {
            "next_step": "exit"
        }

    print("\nTask received:")
    print(task)

    return {
        "messages": [HumanMessage(content=task)],
        "next_step": "developer",
        "code": None,
        "report": None,
        "retry_count": 0,
        "test_cases": None
    }


# ============================================================
# 6. DEVELOPER AGENT
# ============================================================

def real_time_developer(state: CrewState):

    print("\n")
    print("=" * 60)
    print("--- DEVELOPER AGENT ---")
    print("=" * 60)

    task = state["messages"][-1].content

    previous_code = state.get("code")
    previous_report = state.get("report")
    retry_count = state.get("retry_count", 0)

    # --------------------------------------------------------
    # FIRST ATTEMPT
    # --------------------------------------------------------

    if not previous_code or not previous_report:

        print("\n[Developer] Creating solution...")

        dev_prompt = f"""
You are a Senior Python Developer.

Write a clean Python program to solve this coding task:

{task}

Requirements:

1. Read all required values using input().
2. Do not display input prompts.
3. Use input() instead of hardcoded values.
4. Print only the final required output.
5. Do not hardcode the answer.
6. Make the program executable from the terminal.
7. Handle appropriate edge cases.
8. Only return Python code.
9. Do not include explanations.
10. Do not include markdown.
11. Do not use:
    input("Enter something:")
12. Use:
    input()

Example:

Correct:
num = int(input())

Incorrect:
num = int(input("Enter a number: "))
"""

    # --------------------------------------------------------
    # RETRY ATTEMPT
    # --------------------------------------------------------

    else:

        print(
            f"\n[Developer] Fixing code "
            f"(Correction Attempt {retry_count})..."
        )

        dev_prompt = f"""
You are a Senior Python Developer debugging an existing program.

ORIGINAL CODING TASK:

{task}


PREVIOUS CODE:

{previous_code}


TESTER REPORT:

{previous_report}


The Tester found failing test cases.

Your job is to FIX the existing code.

Requirements:

1. Analyze every failed test case.
2. Identify the actual programming mistake.
3. Modify the previous code instead of blindly rewriting it.
4. Preserve correct functionality.
5. Make the code pass all valid test cases.
6. Read input using input().
7. Do not display input prompts.
8. Do not hardcode answers.
9. Handle appropriate edge cases.
10. Print only the required output.
11. Return ONLY Python code.
12. Do not include explanations.
13. Do not include markdown.
14. Do not include ```python.

Return the corrected complete Python program.
"""

    if llm_flash is None:

        raise ValueError(
            "LLM is not initialized. Please configure GEMINI_API_KEY."
        )

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        code_str = (
            content[0].get("text", "")
            if isinstance(content[0], dict)
            else str(content[0])
        )

    else:

        code_str = str(content)

    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print("\nGenerated Code:")
    print("-" * 60)
    print(code_str)
    print("-" * 60)

    return {
        "code": code_str
    }


# ============================================================
# 7. TESTER AGENT
# ============================================================

def real_time_tester(state: CrewState):

    print("\n")
    print("=" * 60)
    print("--- TESTER AGENT ---")
    print("=" * 60)

    task = state["messages"][-1].content
    code = state["code"]

    existing_test_cases = state.get("test_cases")

    try:

        # ====================================================
        # FIRST TESTING CYCLE
        # ====================================================

        if not existing_test_cases:

            print(
                "\n[Tester] Analyzing task independently..."
            )

            print(
                "[Tester] Generating dynamic test cases..."
            )

            test_cases_json = generate_test_cases.invoke({
                "task_description": task,
                "code": code
            })

            cases_str = str(test_cases_json).strip()

            cases_str = (
                cases_str
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            test_cases_list = json.loads(cases_str)

            if not isinstance(test_cases_list, list):

                raise ValueError(
                    "Tester did not return a JSON list."
                )

            saved_test_cases = json.dumps(
                test_cases_list,
                indent=2
            )

            print(
                f"\n[Tester] Generated "
                f"{len(test_cases_list)} dynamic test cases."
            )

        # ====================================================
        # RETRY TESTING CYCLE
        # ====================================================

        else:

            print(
                "\n[Tester] Reusing previously generated "
                "test cases."
            )

            test_cases_list = json.loads(
                existing_test_cases
            )

            saved_test_cases = existing_test_cases

            print(
                f"[Tester] Running the SAME "
                f"{len(test_cases_list)} test cases "
                f"against corrected code."
            )

        # ====================================================
        # EXECUTE TEST CASES
        # ====================================================

        report = ""

        passed = 0
        failed = 0

        for i, test in enumerate(
            test_cases_list,
            1
        ):

            test_input = str(
                test.get("input", "")
            )

            expected = str(
                test.get("expected", "")
            )

            print(
                f"\n[Tester] Running Test Case {i}..."
            )

            actual = run_python_code.invoke({
                "code": code,
                "test_input": test_input
            })

            actual = str(actual)
            expected = str(expected)

            if actual.strip() == expected.strip():

                status = "PASS"
                passed += 1

            else:

                status = "FAIL"
                failed += 1

            print(
                f"Test Case {i}: {status}"
            )

            report += f"""
Test Case {i}: {status}

Input:
{test_input}

Expected:
{expected}

Actual:
{actual}

-------------------------
"""

        # ====================================================
        # FINAL TEST REPORT
        # ====================================================

        final_report = f"""
### TEST RESULTS:

{report}

### SUMMARY:

Total Tests: {len(test_cases_list)}
Passed: {passed}
Failed: {failed}

### TEST SCENARIOS:

{saved_test_cases}
"""

        print("\nTester Report:")
        print(final_report)

        return {
            "report": final_report,
            "test_cases": saved_test_cases
        }

    # ========================================================
    # INVALID JSON
    # ========================================================

    except json.JSONDecodeError:

        report = f"""
Test execution error:

Tester returned invalid JSON.

Tester Response:
{existing_test_cases}
"""

        print(report)

        return {
            "report": report
        }

    # ========================================================
    # OTHER TESTING ERROR
    # ========================================================

    except Exception:

        report = (
            "Test execution error:\n"
            + traceback.format_exc()
        )

        print(report)

        return {
            "report": report
        }


# ============================================================
# 8. MANAGER AGENT
# ============================================================

def manager_decision_node(state: CrewState):

    print("\n")
    print("=" * 60)
    print("--- MANAGER AGENT ---")
    print("=" * 60)

    report = state.get(
        "report",
        "No report available."
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    print("\nTEST REPORT:")
    print(report)

    manager_prompt = f"""
You are the Manager Agent in an automated software development system.

Analyze this Tester Report:

{report}

Rules:

- If every test case has PASS status, return PASS.
- If even one test case has FAIL status, return RETRY.
- If there is a test execution error, return RETRY.

Return ONLY:

PASS

or

RETRY
"""

    try:

        response = llm.invoke(manager_prompt)

        decision = str(
            response.content
            if hasattr(response, "content")
            else response
        ).strip().upper()

    except Exception:

        decision = "RETRY"

    failed_tests = re.findall(
        r"Test Case \d+:\s*FAIL",
        report
    )

    execution_error = (
        "Test execution error" in report
        or "Execution Error" in report
    )

    if failed_tests or execution_error:

        decision = "RETRY"

    else:

        decision = "PASS"

    print(
        "\n[Manager] Decision:",
        decision
    )

    # ========================================================
    # ALL TESTS PASSED
    # ========================================================

    if decision == "PASS":

        print(
            "\n[Manager] All tests passed."
        )

        while True:

            user_choice = input(
                "\nWhat would you like to do? "
                "(another / store / exit): "
            ).lower().strip()

            if user_choice == "another":

                print(
                    "\n[Manager] Starting another coding task..."
                )

                return {
                    "next_step": "task_input",
                    "retry_count": 0,
                    "code": None,
                    "report": None,
                    "test_cases": None
                }

            elif user_choice == "store":

                print(
                    "\n[Manager] Sending validated code "
                    "to Archiver."
                )

                return {
                    "next_step": "archiver"
                }

            elif user_choice == "exit":

                print(
                    "\n[Manager] Exiting workflow."
                )

                return {
                    "next_step": "exit"
                }

            else:

                print(
                    "\nInvalid option."
                    "\nPlease enter: another, store, or exit."
                )

    # ========================================================
    # TESTS FAILED
    # ========================================================

    MAX_RETRIES = 3

    if retry_count < MAX_RETRIES:

        new_retry_count = retry_count + 1

        print(
            "\n[Manager] Tests failed."
        )

        print(
            "[Manager] Sending code back to Developer."
        )

        print(
            f"[Manager] Correction Attempt: "
            f"{new_retry_count}/{MAX_RETRIES}"
        )

        return {
            "next_step": "developer",
            "retry_count": new_retry_count
        }

    # ========================================================
    # MAXIMUM RETRIES
    # ========================================================

    print(
        "\n[Manager] Maximum correction attempts reached."
    )

    print(
        "[Manager] Automatic correction failed."
    )

    return {
        "next_step": "failed"
    }


# ============================================================
# 9. ARCHIVER AGENT
# ============================================================

def archiver_node(state: CrewState):

    print("\n")
    print("=" * 60)
    print("--- ARCHIVER AGENT ---")
    print("=" * 60)

    code = state.get(
        "code",
        ""
    )

    report = state.get(
        "report",
        ""
    )

    print(
        "\n[Archiver] Final validated code:"
    )

    print("-" * 60)
    print(code)
    print("-" * 60)

    print(
        "\n[Archiver] Final test report:"
    )

    print(report)

    print(
        "\n[Archiver] Code successfully validated and stored."
    )

    return {
        "next_step": "complete"
    }


# ============================================================
# 10. FAILURE AGENT
# ============================================================

def failure_node(state: CrewState):

    print("\n")
    print("=" * 60)
    print("--- MANAGER FAILURE REPORT ---")
    print("=" * 60)

    print(
        "\nAutomatic correction failed after "
        "maximum retry attempts."
    )

    print("\nLast generated code:")

    print("-" * 60)

    print(
        state.get(
            "code",
            "No code available."
        )
    )

    print("-" * 60)

    print("\nFinal test report:")

    print(
        state.get(
            "report",
            "No report available."
        )
    )

    print(
        "\n[System] Task stopped for manual review."
    )

    return {
        "next_step": "failed"
    }


# ============================================================
# 11. LANGGRAPH WORKFLOW
# ============================================================

rt_workflow = StateGraph(
    CrewState
)


rt_workflow.add_node(
    "task_input",
    task_input_node
)


rt_workflow.add_node(
    "developer",
    real_time_developer
)


rt_workflow.add_node(
    "tester",
    real_time_tester
)


rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)


rt_workflow.add_node(
    "archiver",
    archiver_node
)


rt_workflow.add_node(
    "failure",
    failure_node
)


# ============================================================
# START → TASK INPUT
# ============================================================

rt_workflow.add_edge(
    START,
    "task_input"
)


# ============================================================
# TASK INPUT → DEVELOPER
# ============================================================

def route_from_input(state):

    if state.get("next_step") == "exit":

        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# ============================================================
# DEVELOPER → TESTER
# ============================================================

rt_workflow.add_edge(
    "developer",
    "tester"
)


# ============================================================
# TESTER → MANAGER
# ============================================================

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# ============================================================
# MANAGER ROUTING
# ============================================================

def route_from_decision(state):

    next_step = state.get(
        "next_step"
    )

    if next_step == "archiver":

        return "archiver"

    if next_step == "developer":

        return "developer"

    if next_step == "task_input":

        return "task_input"

    if next_step == "failed":

        return "failure"

    if next_step == "exit":

        return END

    return END


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ============================================================
# ARCHIVER → END
# ============================================================

rt_workflow.add_edge(
    "archiver",
    END
)


# ============================================================
# FAILURE → END
# ============================================================

rt_workflow.add_edge(
    "failure",
    END
)


# ============================================================
# 12. COMPILE GRAPH
# ============================================================

rt_app = rt_workflow.compile()


# ============================================================
# 13. RUN WORKFLOW
# ============================================================

if __name__ == "__main__":

    try:

        rt_app.invoke(
            {
                "messages": [],
                "next_step": None,
                "code": None,
                "report": None,
                "retry_count": 0,
                "test_cases": None
            },
            config={
                "recursion_limit": 50
            }
        )

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as e:

        print(
            f"\nAn error occurred: {e}"
        )
