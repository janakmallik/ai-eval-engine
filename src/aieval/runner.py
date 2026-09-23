from collections.abc import Callable, Iterable

from aieval.dataset import EvalCase
from aieval.evaluators.base import Evaluator
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


def evaluate_dataset(
    model: Callable[[str], str],
    dataset: Iterable[EvalCase],
    evaluator: Evaluator,
) -> EvaluationRun:

    results: list[EvaluationResult] = []

    for case in dataset:
        actual = model(case.input)

        result = evaluator.evaluate(
            case_id=case.id,
            expected=case.expected,
            actual=actual,
        )

        results.append(result)

    return EvaluationRun(results)
