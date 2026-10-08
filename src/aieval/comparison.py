from dataclasses import dataclass, field

from aieval.experiment import Experiment
from aieval.run import EvaluationRun


@dataclass
class ComparisonResult:
    baseline_score: float
    current_score: float
    score_delta: float
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float

    baseline_latency: float = 0.0
    current_latency: float = 0.0
    latency_delta: float = 0.0

    baseline_cost: float = 0.0
    current_cost: float = 0.0
    cost_delta: float = 0.0

    baseline_error_rate: float = 0.0
    current_error_rate: float = 0.0
    error_rate_delta: float = 0.0

    evaluator_deltas: dict[str, float] = field(default_factory=dict)
    added_evaluators: set[str] = field(default_factory=set)
    removed_evaluators: set[str] = field(default_factory=set)
    model_changed: bool = False
    prompt_changed: bool = False
    dataset_changed: bool = False
    evaluator_config_changed: bool = False
    model_version_changed: bool = False
    prompt_version_changed: bool = False
    dataset_version_changed: bool = False


# Step by step:
# baseline.summaries() — calls a method on baseline that presumably returns some iterable (list, dict, etc.) of "summary" objects.
# set(...) — converts that iterable into a set. Sets store unique elements and have no order.
# | — the set union operator. A | B returns a new set containing everything in A or B (duplicates removed).
# set(current.summaries()) — same conversion for the other run.
# evaluator_names — the combined set of every evaluator name that appears in either run.
# Why union? Because you want to compare all evaluators — even ones that only exist in one of the two runs. If an evaluator was added in current, or removed since baseline, you still want to detect it.
def compare_runs(
    baseline: EvaluationRun,
    current: EvaluationRun,
) -> ComparisonResult:
    baseline_evaluators = set(baseline.summaries())
    current_evaluators = set(current.summaries())

    evaluator_names = baseline_evaluators & current_evaluators

    added_evaluators = current_evaluators - baseline_evaluators
    removed_evaluators = baseline_evaluators - current_evaluators

    evaluator_deltas = {}

    for evaluator_name in evaluator_names:
        baseline_score = baseline.summary(evaluator_name).score
        current_score = current.summary(evaluator_name).score

        evaluator_deltas[evaluator_name] = current_score - baseline_score

    baseline_latency = (
        baseline.trace.duration
        if baseline.trace is not None
        else float(baseline.metadata.get("latency", 0.0))
    )
    current_latency = (
        current.trace.duration
        if current.trace is not None
        else float(current.metadata.get("latency", 0.0))
    )

    baseline_cost = (
        baseline.trace.summary()["total_cost"]
        if baseline.trace is not None
        else float(baseline.metadata.get("cost", 0.0))
    )
    current_cost = (
        current.trace.summary()["total_cost"]
        if current.trace is not None
        else float(current.metadata.get("cost", 0.0))
    )

    baseline_error_rate = float(baseline.metadata.get("error_rate", 0.0))
    current_error_rate = float(current.metadata.get("error_rate", 0.0))

    return ComparisonResult(
        baseline_score=baseline.score,
        current_score=current.score,
        score_delta=current.score - baseline.score,
        baseline_pass_rate=baseline.pass_rate,
        current_pass_rate=current.pass_rate,
        pass_rate_delta=current.pass_rate - baseline.pass_rate,
        baseline_latency=baseline_latency,
        current_latency=current_latency,
        latency_delta=current_latency - baseline_latency,
        baseline_cost=baseline_cost,
        current_cost=current_cost,
        cost_delta=current_cost - baseline_cost,
        baseline_error_rate=baseline_error_rate,
        current_error_rate=current_error_rate,
        error_rate_delta=current_error_rate - baseline_error_rate,
        evaluator_deltas=evaluator_deltas,
        added_evaluators=added_evaluators,
        removed_evaluators=removed_evaluators,
    )


def compare_experiments(
    baseline: Experiment,
    current: Experiment,
) -> ComparisonResult:
    comparison = compare_runs(
        baseline.run,
        current.run,
    )

    comparison.model_changed = baseline.model != current.model
    comparison.prompt_changed = baseline.prompt != current.prompt
    comparison.dataset_changed = baseline.dataset != current.dataset
    comparison.evaluator_config_changed = (
        baseline.evaluator_config != current.evaluator_config
    )

    comparison.model_version_changed = baseline.model_version != current.model_version
    comparison.prompt_version_changed = (
        baseline.prompt_version != current.prompt_version
    )
    comparison.dataset_version_changed = (
        baseline.dataset_version != current.dataset_version
    )

    return comparison
