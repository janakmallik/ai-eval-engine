from aieval.comparison import ComparisonResult
from aieval.regression import RegressionDetector


def test_regression_detector_detects_score_regression():
    comparison = ComparisonResult(
        baseline_score=0.92,
        current_score=0.86,
        score_delta=-0.06,
        baseline_pass_rate=0.92,
        current_pass_rate=0.86,
        pass_rate_delta=-0.06,
        evaluator_deltas={
            "exact_match": -0.10,
            "similarity": 0.01,
        },
    )

    detector = RegressionDetector(threshold=0.05)

    result = detector.check(comparison)

    assert result.regressed is True
    assert result.score_regression is True
    assert result.evaluator_regressions == ["exact_match"]


def test_regression_detector_allows_small_changes():
    comparison = ComparisonResult(
        baseline_score=0.92,
        current_score=0.90,
        score_delta=-0.02,
        baseline_pass_rate=0.92,
        current_pass_rate=0.90,
        pass_rate_delta=-0.02,
        evaluator_deltas={
            "exact_match": -0.02,
            "similarity": 0.01,
        },
    )

    detector = RegressionDetector(threshold=0.05)

    result = detector.check(comparison)

    assert result.regressed is False
    assert result.score_regression is False
    assert result.evaluator_regressions == []