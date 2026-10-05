from collections.abc import Callable, Iterable

from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.base import Evaluator
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun
from aieval.tracing.response import ModelResponse
from aieval.tracing.trace import Trace


def evaluate_dataset(
    model: Callable[[str], str],
    dataset: Iterable[EvalCase],
    evaluators: Iterable[Evaluator],
    metadata: dict[str, object] | None = None,
    enable_tracing: bool = False,
    retriever: Callable[[str, int], Iterable[object]] | None = None,
    retrieval_top_k: int = 5,
    tool: Callable[[str], Iterable[object]] | None = None,
    model_name: str = "model",
    model_provider: str = "unknown",
) -> EvaluationRun:

    results: list[EvaluationResult] = []
    total_cases = 0
    passed_cases = 0
    failed_cases = 0
    trace = Trace() if enable_tracing else None
    root_span = trace.start_span("evaluation") if trace else None

    try:
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

            if retriever is not None:
                retrieval_span = (
                    trace.start_retrieval(
                        query=case.input,
                        top_k=retrieval_top_k,
                        parent_span_id=case_span.span_id,
                    )
                    if trace and case_span
                    else None
                )

                if retrieval_span:
                    with retrieval_span:
                        retrieved = retriever(
                            case.input,
                            retrieval_top_k,
                        )
                        retrieval_span.record_retrieval_result(
                            result_count=len(retrieved),
                        )
                else:
                    retrieved = retriever(
                        case.input,
                        retrieval_top_k,
                    )

            if tool is not None:
                tool_span = (
                    trace.start_tool(
                        tool="tool",
                        parent_span_id=case_span.span_id,
                    )
                    if trace and case_span
                    else None
                )

                if tool_span:
                    tool_span.set_attribute("tool.input", case.input)

                    with tool_span:
                        tool_result = tool(case.input)

                        tool_span.set_attribute("tool.output", tool_result)

                        tool_span.record_tool_result(
                            result_count=len(tool_result),
                        )
                else:
                    tool_result = tool(case.input)

            try:
                model_span = (
                    trace.start_model(
                        model=model_name,
                        provider=model_provider,
                        parent_span_id=case_span.span_id,
                    )
                    if trace and case_span
                    else None
                )

                if retriever is not None:
                    model_input = "\n\n".join(
                        [case.input, *[str(document) for document in retrieved]]
                    )
                else:
                    model_input = case.input

                if model_span:
                    model_span.set_attribute("model.input", model_input)

                    with model_span:
                        response = model(model_input)

                    if isinstance(response, ModelResponse):
                        actual = response.output

                        if response.usage is not None:
                            model_span.record_usage(response.usage)

                        if response.finish_reason is not None:
                            model_span.record_finish_reason(response.finish_reason)

                        if response.response_id is not None:
                            model_span.record_response_id(response.response_id)

                        if response.request_id is not None:
                            model_span.record_request_id(response.request_id)

                        if response.temperature is not None:
                            model_span.record_temperature(response.temperature)

                        if response.max_tokens is not None:
                            model_span.record_max_tokens(response.max_tokens)

                    else:
                        actual = response

                    model_span.set_attribute("model.output", actual)
                else:
                    response = model(model_input)
                    actual = (
                        response.output
                        if isinstance(response, ModelResponse)
                        else response
                    )

                context = EvaluationContext(
                    case=case,
                    actual=actual,
                    retrieved=retrieved if retriever is not None else None,
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
                        if evaluator_span:
                            with evaluator_span:
                                result = evaluator.evaluate(context)

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
                        else:
                            result = evaluator.evaluate(context)

                        results.append(result)
                        case_passed = case_passed and result.passed

                    finally:
                        if evaluator_span and evaluator_span.ended_at is None:
                            evaluator_span.end()

                if case_passed:
                    passed_cases += 1
                else:
                    failed_cases += 1

            finally:
                if case_span and case_span.ended_at is None:
                    case_span.set_attribute(
                        "evaluation.passed",
                        case_passed,
                    )
                    case_span.end()

    finally:
        if root_span and root_span.ended_at is None:
            total_results = len(results)
            passed_results = sum(result.passed for result in results)
            failed_results = total_results - passed_results
            pass_rate = passed_results / total_results if total_results else 0.0

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
