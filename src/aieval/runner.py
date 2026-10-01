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
    trace = Trace() if enable_tracing else None
    root_span = trace.start_span("evaluation") if trace else None

    for case in dataset:
        case_span = (
            trace.start_span(
                "evaluation.case",
                parent_span_id=root_span.span_id,
            )
            if trace and root_span
            else None
        )

        model_span = (
            trace.start_span(
                "model",
                parent_span_id=case_span.span_id,
            )
            if trace and case_span
            else None
        )

        if model_span:
            with model_span:
                actual = model(case.input)
        else:
            actual = model(case.input)

        context = EvaluationContext(
            case=case,
            actual=actual,
        )

        for evaluator in evaluators:
            evaluator_span = (
                trace.start_span(
                    "evaluator",
                    parent_span_id=case_span.span_id,
                )
                if trace and case_span
                else None
            )

            try:
                result = evaluator.evaluate(context)
                results.append(result)
            finally:
                if evaluator_span:
                    evaluator_span.end()

        if case_span:
            case_span.end()

    if root_span:
        root_span.end()

    return EvaluationRun(
        results,
        metadata=metadata or {},
        trace=trace,
    )