from dataclasses import dataclass

from aieval.run import EvaluationRun


@dataclass
class ComparisonResult:
    baseline_score: float
    current_score: float
    score_delta: float
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    evaluator_deltas: dict[str, float]

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
    evaluator_names = set(baseline.summaries()) | set(current.summaries())

# {} creates an empty dictionary.
# you'll do something like evaluator_deltas[name] = current_score - baseline_score to record the change for each evaluator.
    evaluator_deltas = {}

    for evaluator_name in evaluator_names:
        baseline_score = baseline.summary(evaluator_name).score
        current_score = current.summary(evaluator_name).score

        evaluator_deltas[evaluator_name] = current_score - baseline_score

    return ComparisonResult(
        baseline_score=baseline.score,
        current_score=current.score,
        score_delta=current.score - baseline.score,
        baseline_pass_rate=baseline.pass_rate,
        current_pass_rate=current.pass_rate,
        pass_rate_delta=current.pass_rate - baseline.pass_rate,
        evaluator_deltas=evaluator_deltas,
    )