from pathlib import Path
from runpy import run_path

DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "rag_demo.py"
run_demo = run_path(str(DEMO_PATH))["run_demo"]


def test_rag_demo_returns_answer_and_retrieved_documents():
    rag_result, _ = run_demo()

    assert rag_result.question == "What is the capital of France?"
    assert rag_result.answer == "Paris is the capital of France."
    assert rag_result.documents == [
        "Paris is the capital of France.",
        "France is a country in Europe.",
    ]


def test_rag_demo_evaluators_pass_for_supported_answer():
    _, results = run_demo()

    assert len(results) == 3
    assert all(result.passed for result in results)


def test_rag_demo_includes_all_three_rag_evaluators():
    _, results = run_demo()

    assert {result.evaluator_name for result in results} == {
        "context_relevance",
        "answer_relevance",
        "faithfulness",
    }
