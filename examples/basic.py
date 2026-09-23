from aieval.dataset import EvalCase
from aieval.evaluator import ExactMatchEvaluator
from aieval.runner import evaluate_dataset

dataset = [
    EvalCase(
        id="001",
        input="What is 2 + 2?",
        expected="4",
    ),
    EvalCase(
        id="002",
        input="What is the capital of France?",
        expected="Paris",
    ),
    EvalCase(
        id="003",
        input="What is 10 / 2?",
        expected="5",
    ),
]


def fake_model(question: str) -> str:
    answers = {
        "What is 2 + 2?": "4",
        "What is the capital of France?": "Paris",
        "What is 10 / 2?": "6",
    }

    return answers[question]


results = evaluate_dataset(
    model=fake_model,
    dataset=dataset,
    evaluator=ExactMatchEvaluator(),
)

for result in results:
    print(result)
