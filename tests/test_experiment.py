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

def test_experiment_accepts_metadata():
    run = EvaluationRun(results=[])

    experiment = Experiment(
        name="baseline-v1",
        model="test-model",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        metadata={
            "model_version": "1.0",
            "prompt_version": "v3",
            "dataset_version": "v2",
        },
    )

    assert experiment.metadata == {
        "model_version": "1.0",
        "prompt_version": "v3",
        "dataset_version": "v2",
    }

def test_experiment_metadata_defaults_to_independent_dicts():
    run = EvaluationRun(results=[])

    first = Experiment(
        name="experiment-1",
        model="test-model",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    second = Experiment(
        name="experiment-2",
        model="test-model",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    first.metadata["model_version"] = "1.0"

    assert first.metadata == {"model_version": "1.0"}
    assert second.metadata == {}