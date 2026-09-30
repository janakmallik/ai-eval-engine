from aieval.experiment import Experiment
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


def test_experiment_stores_evaluation_context():
    run = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="4",
                actual="4",
                score=1.0,
                passed=True,
            )
        ]
    )

    experiment = Experiment(
        name="baseline-v1",
        model="test-model",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    assert experiment.name == "baseline-v1"
    assert experiment.model == "test-model"
    assert experiment.prompt == "Answer the question."
    assert experiment.dataset == "math-v1"
    assert experiment.run is run

def test_experiment_to_dict():
    run = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="4",
                actual="4",
                score=1.0,
                passed=True,
            )
        ]
    )

    experiment = Experiment(
        name="baseline-v1",
        model="test-model",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    data = experiment.to_dict()

    assert data["name"] == "baseline-v1"
    assert data["model"] == "test-model"
    assert data["prompt"] == "Answer the question."
    assert data["dataset"] == "math-v1"
    assert data["run"] == run.to_dict()