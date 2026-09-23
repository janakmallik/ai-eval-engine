from collections.abc import Callable, Iterable

from aieval.dataset import EvalCase
from aieval.evaluator import ExactMatchEvaluator
from aieval.result import EvaluationResult


def evaluate_dataset(
    model: Callable[[str], str],
    dataset: Iterable[EvalCase],
    evaluator: ExactMatchEvaluator,
) -> list[EvaluationResult]:

    results = []

    for case in dataset:
        actual = model(case.input)

        result = evaluator.evaluate(
            case_id=case.id,
            expected=case.expected,
            actual=actual,
        )

        results.append(result)

    return results
