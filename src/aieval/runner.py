from collections.abc import Callable, Iterable

from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.base import Evaluator
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun
from aieval.tracing.trace import Trace

def evaluate_dataset(
    model: Callable[[str], str],
    dataset: Iterable[EvalCase],
    evaluators: Iterable[Evaluator],
    metadata: dict[str, object] | None = None,
    enable_tracing: bool = False,
) -> EvaluationRun:

    results: list[EvaluationResult] = []
    total_cases = 0
    passed_cases = 0
    failed_cases = 0
    trace = Trace() if enable_tracing else None
    root_span = trace.start_span("evaluation") if trace else None

    for case in dataset:
        total_cases += 1
        case_passed = True

        case_span = (
            trace.start_span(
                "evaluation.case",
                parent_span_id=root_span.span_id,
            )
            if trace and root_span
            else None
        )

        if case_span:
            case_span.set_attribute("case.id", case.id)
            case_span.set_attribute("case.input", case.input)
            case_span.set_attribute("case.expected", case.expected)

        model_span = (
            trace.start_span(
                "model",
                parent_span_id=case_span.span_id,
            )
            if trace and case_span
            else None
        )

        if model_span:
            model_span.set_attribute("model.input", case.input)

            with model_span:
                actual = model(case.input)

            model_span.set_attribute("model.output", actual)
        else:
            actual = model(case.input)

        context = EvaluationContext(
            case=case,
            actual=actual,
        )

        for evaluator in evaluators:
            evaluator_name = getattr(
                evaluator,
                "name",
                evaluator.__class__.__name__.removesuffix("Evaluator").lower(),
            )

            evaluator_span = (
                trace.start_span(
                    f"evaluator.{evaluator_name}",
                    parent_span_id=case_span.span_id,
                )
                if trace and case_span
                else None
            )

            try:
                result = evaluator.evaluate(context)
                results.append(result)

                case_passed = case_passed and result.passed

                if evaluator_span:
                    evaluator_span.set_attribute(
                        "evaluator.name",
                        evaluator_name,
                    )
                    evaluator_span.set_attribute(
                        "evaluation.case_id",
                        result.case_id,
                    )
                    evaluator_span.set_attribute(
                        "evaluation.score",
                        result.score,
                    )
                    evaluator_span.set_attribute(
                        "evaluation.passed",
                        result.passed,
                    )
            finally:
                if evaluator_span:
                    evaluator_span.end()

        if case_passed:
            passed_cases += 1
        else:
            failed_cases += 1

        if case_span:
            case_span.set_attribute(
                "evaluation.passed",
                case_passed,
            )
            case_span.end()

    if root_span:
        total_results = len(results)
        passed_results = sum(result.passed for result in results)
        failed_results = total_results - passed_results
        pass_rate = (
            passed_results / total_results
            if total_results
            else 0.0
        )

        root_span.set_attribute(
            "evaluation.total_cases",
            total_cases,
        )
        root_span.set_attribute(
            "evaluation.passed_cases",
            passed_cases,
        )
        root_span.set_attribute(
            "evaluation.failed_cases",
            failed_cases,
        )
        root_span.set_attribute(
            "evaluation.total_results",
            total_results,
        )
        root_span.set_attribute(
            "evaluation.passed",
            passed_results,
        )
        root_span.set_attribute(
            "evaluation.failed",
            failed_results,
        )
        root_span.set_attribute(
            "evaluation.passed_results",
            passed_results,
        )
        root_span.set_attribute(
            "evaluation.failed_results",
            failed_results,
        )
        root_span.set_attribute(
            "evaluation.pass_rate",
            pass_rate,
        )

        root_span.end()

    return EvaluationRun(
        results,
        metadata=metadata or {},
        trace=trace,
    )