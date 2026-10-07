from aieval import (
    EvalCase,
    ExactMatchEvaluator,
    evaluate_dataset,
)
from aieval.reporting.json import JsonReporter

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


def model_v1(prompt: str) -> str:
    answers = {
        "What is the capital of France?": "Paris",
        "What is the capital of Italy?": "Rome",
    }

    return answers[prompt]


def model_v2(prompt: str) -> str:
    answers = {
        "What is the capital of France?": "Paris",
        "What is the capital of Italy?": "Milan",
    }

    return answers[prompt]


evaluator = ExactMatchEvaluator()
reporter = JsonReporter()


baseline = evaluate_dataset(
    model=model_v1,
    dataset=dataset,
    evaluators=[evaluator],
    metadata={
        "model": "model_v1",
        "dataset": "capitals-v1",
    },
)

current = evaluate_dataset(
    model=model_v2,
    dataset=dataset,
    evaluators=[evaluator],
    metadata={
        "model": "model_v2",
        "dataset": "capitals-v1",
    },
)


reporter.write(baseline, "baseline.json")
reporter.write(current, "current.json")

print("Created baseline.json and current.json")
