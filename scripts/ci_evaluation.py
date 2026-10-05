# a deterministic CI evaluation script
# This script should:
# - use a tiny deterministic dataset
# - use a deterministic fake model
# - run evaluate_dataset()
# - write current.json
# - not intentionally regress

from pathlib import Path

from aieval.dataset import EvalCase
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.reporting.json import JsonReporter
from aieval.runner import evaluate_dataset

dataset = [
    EvalCase(
        id="1",
        input="What is the capital of France?",
        expected="Paris",
    ),
    EvalCase(
        id="2",
        input="What is the capital of Italy?",
        expected="Rome",
    ),
]


def model(prompt: str) -> str:
    answers = {
        "What is the capital of France?": "Paris",
        "What is the capital of Italy?": "Rome",
    }

    return answers[prompt]


run = evaluate_dataset(
    model=model,
    dataset=dataset,
    evaluators=[ExactMatchEvaluator()],
    metadata={
        "model": "ci-model",
        "dataset": "capitals-v1",
    },
)

JsonReporter().write(
    run,
    Path("current.json"),
)
