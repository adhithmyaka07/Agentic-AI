import os
import io
import sys
import traceback

from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0
)


class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output or error."""

    clean_code = (
        str(code)
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(clean_code, {}, {})
        result = new_stdout.getvalue()

    except Exception:
        result = traceback.format_exc()

    finally:
        sys.stdout = old_stdout

    return result.strip() or "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate test cases for a coding task."""

    prompt = f"""
You are a senior QA engineer.

Generate 3 to 5 specific test cases for:

{task_description}

Include normal cases, edge cases and invalid cases.

Return a numbered list.
"""

    response = llm.invoke(prompt)

    return str(response.content)


def developer_node(state: CrewState):

    task = state["messages"][-1].content

    prompt = f"""
Write a clean Python program to solve this task:

{task}

Return ONLY Python code.
Do not use Markdown.
Do not include ```python.
"""

    response = llm.invoke(prompt)

    code = str(response.content)

    code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code
    }


def tester_node(state: CrewState):

    task = state["messages"][-1].content

    tests = generate_test_cases.invoke({
        "task_description": task
    })

    execution = run_python_code.invoke({
        "code": state["code"]
    })

    report = f"""
### GENERATED CODE

{state["code"]}

### EXECUTION OUTPUT

{execution}

### TEST CASES

{tests}
"""

    return {
        "report": report
    }


workflow = StateGraph(CrewState)

workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

graph = workflow.compile()


def run_langgraph(task: str):

    result = graph.invoke({
        "messages": [
            HumanMessage(content=task)
        ],
        "code": None,
        "report": None
    })

    return result["report"]