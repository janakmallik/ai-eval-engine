from aieval import (
    ContainsEvaluator,
    EvalCase,
    EvalDataset,
    EvaluationContext,
    EvaluationResult,
    EvaluationRun,
    ExactMatchEvaluator,
    LengthEvaluator,
    PerformanceMetrics,
    SimilarityEvaluator,
    evaluate_dataset,
)


def test_public_api_exposes_regression_components():
    import aieval

    assert aieval.ComparisonResult
    assert aieval.RegressionConfig
    assert aieval.RegressionDetector
    assert aieval.RegressionResult
    assert aieval.RegressionGate
    assert aieval.GateResult


def test_public_api_exports_core_v1_objects():
    assert EvalCase is not None
    assert EvalDataset is not None
    assert EvaluationContext is not None
    assert EvaluationResult is not None
    assert EvaluationRun is not None

    assert ExactMatchEvaluator is not None
    assert ContainsEvaluator is not None
    assert SimilarityEvaluator is not None
    assert LengthEvaluator is not None

    assert evaluate_dataset is not None


def test_public_api_exposes_tracing_components():
    import aieval

    assert aieval.Trace
    assert aieval.Span
    assert aieval.ModelUsage
    assert aieval.ModelResponse
    assert aieval.ToolResponse


def test_public_api_exposes_evaluate_with_config():
    import aieval

    assert aieval.evaluate_with_config


def test_performance_metrics_is_public():
    assert PerformanceMetrics is not None


def test_answer_relevance_evaluator_is_exported_from_aieval():
    from aieval import AnswerRelevanceEvaluator

    assert AnswerRelevanceEvaluator.__name__ == "AnswerRelevanceEvaluator"


def test_context_relevance_evaluator_is_exported_from_aieval():
    from aieval import ContextRelevanceEvaluator

    assert ContextRelevanceEvaluator.__name__ == "ContextRelevanceEvaluator"


def test_faithfulness_evaluator_is_exported_from_aieval():
    from aieval import FaithfulnessEvaluator

    assert FaithfulnessEvaluator.__name__ == "FaithfulnessEvaluator"
