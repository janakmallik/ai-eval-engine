from aieval.dataset import EvalCase
from aieval.evaluators.contains import ContainsEvaluator
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.runner import evaluate_dataset
from aieval.evaluators.length import LengthEvaluator

def test_evaluate_dataset():

    dataset = [
        EvalCase(
            id="001",
            input="What is 2 + 2?",
            expected="4",
        ),
        EvalCase(
            id="002",
            input="What is 3 + 3?",
            expected="6",
        ),
    ]

    def fake_model(question: str) -> str:
        answers = {
            "What is 2 + 2?": "4",
            "What is 3 + 3?": "6",
        }

        return answers[question]

    results = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    assert results.total == 2
    assert results.passed == 2
    assert results.failed == 0
    assert results.score == 1.0

    assert results.results[0].score == 1.0
    assert results.results[0].passed is True

    assert results.results[1].score == 1.0
    assert results.results[1].passed is True


def test_runner_with_contains_evaluator():

    dataset = [
        EvalCase(
            id="001",
            input="What is the capital of France?",
            expected="Paris",
        ),
        EvalCase(
            id="002",
            input="Tell me about Paris.",
            expected="Paris",
        ),
    ]

    def fake_model(question: str) -> str:
        answers = {
            "What is the capital of France?": "Paris is the capital of France.",
            "Tell me about Paris.": "Paris is a city in France.",
        }

        return answers[question]

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ContainsEvaluator()],
    )

    assert run.total == 2
    assert run.passed == 2
    assert run.failed == 0
    assert run.score == 1.0


def test_runner_accepts_different_evaluator():

    dataset = [
        EvalCase(
            id="001",
            input="What is the capital of France?",
            expected="Paris",
        ),
    ]

    def fake_model(question: str) -> str:
        return "Paris is the capital of France."

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ContainsEvaluator()],
    )

    assert run.total == 1
    assert run.passed == 1
    assert run.failed == 0
    assert run.score == 1.0

def test_runner_accepts_multiple_evaluators():

    dataset = [
        EvalCase(
            id="011",
            input="What is the capital of France?",
            expected="Paris",
        ),
    ]

    def fake_model(question: str) -> str:
        return "Paris is the capital of France."

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[
            ContainsEvaluator(),
            LengthEvaluator(max_length=100),
        ],
    )

    assert run.total == 2
    assert run.passed == 2
    assert run.failed == 0