from aieval.experiment import Experiment
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun
from aieval.comparison import compare_experiments

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

def test_compare_experiments():
    baseline_run = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
        ]
    )

    current_run = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="London",
                score=0.0,
                passed=False,
            ),
        ]
    )

    baseline = Experiment(
        name="baseline-v1",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=baseline_run,
    )

    current = Experiment(
        name="current-v2",
        model="model-v2",
        prompt="Answer the question.",
        dataset="math-v1",
        run=current_run,
    )

    comparison = compare_experiments(baseline, current)

    assert comparison.score_delta == -1.0
    assert comparison.pass_rate_delta == -1.0

def test_compare_experiments_detects_no_configuration_changes():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    current = Experiment(
        name="current-v2",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    comparison = compare_experiments(baseline, current)

    assert comparison.model_changed is False
    assert comparison.prompt_changed is False
    assert comparison.dataset_changed is False

def test_experiment_accepts_evaluator_configuration():
    run = EvaluationRun(results=[])

    experiment = Experiment(
        name="baseline-v1",
        model="test-model",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "exact_match": {
                "threshold": 1.0,
            },
            "similarity": {
                "threshold": 0.8,
            },
        },
    )

    assert experiment.evaluator_config["exact_match"]["threshold"] == 1.0
    assert experiment.evaluator_config["similarity"]["threshold"] == 0.8

def test_compare_experiments_tracks_evaluator_configuration_changes():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "similarity": {
                "threshold": 0.8,
            },
        },
    )

    current = Experiment(
        name="current-v2",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "similarity": {
                "threshold": 0.9,
            },
        },
    )

    comparison = compare_experiments(baseline, current)

    assert comparison.evaluator_config_changed is True

def test_experiment_to_dict_includes_configuration_and_run():
    run = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            )
        ]
    )

    experiment = Experiment(
        name="baseline-v1",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "exact_match": {
                "threshold": 1.0,
            }
        },
    )

    data = experiment.to_dict()

    assert data["name"] == "baseline-v1"
    assert data["model"] == "model-v1"
    assert data["prompt"] == "Answer the question."
    assert data["dataset"] == "math-v1"
    assert data["evaluator_config"]["exact_match"]["threshold"] == 1.0
    assert data["run"]["total"] == 1
    assert data["run"]["score"] == 1.0

def test_experiment_has_stable_id():
    run = EvaluationRun(results=[])

    experiment = Experiment(
        name="baseline-v1",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    assert experiment.experiment_id
    assert isinstance(experiment.experiment_id, str)

def test_identical_experiments_have_same_id():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "exact_match": {
                "threshold": 1.0,
            }
        },
    )

    duplicate = Experiment(
        name="another-name",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "exact_match": {
                "threshold": 1.0,
            }
        },
    )

    assert baseline.experiment_id == duplicate.experiment_id

def test_changed_experiment_configuration_has_different_id():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="model-v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "similarity": {
                "threshold": 0.8,
            }
        },
    )

    changed = Experiment(
        name="baseline-v1",
        model="model-v2",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
        evaluator_config={
            "similarity": {
                "threshold": 0.8,
            }
        },
    )

    assert baseline.experiment_id != changed.experiment_id

def test_experiment_accepts_version_metadata():
    run = EvaluationRun(results=[])

    experiment = Experiment(
        name="baseline-v1",
        model="gpt-model",
        model_version="2026-09-30",
        prompt="Answer the question.",
        prompt_version="prompt-v3",
        dataset="math",
        dataset_version="2026-09-29",
        run=run,
    )

    assert experiment.model_version == "2026-09-30"
    assert experiment.prompt_version == "prompt-v3"
    assert experiment.dataset_version == "2026-09-29"

def test_changed_version_metadata_has_different_id():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="gpt-model",
        model_version="v1",
        prompt="Answer the question.",
        prompt_version="v1",
        dataset="math",
        dataset_version="v1",
        run=run,
    )

    changed = Experiment(
        name="baseline-v1",
        model="gpt-model",
        model_version="v2",
        prompt="Answer the question.",
        prompt_version="v1",
        dataset="math",
        dataset_version="v1",
        run=run,
    )

    assert baseline.experiment_id != changed.experiment_id

def test_experiment_to_dict_includes_version_metadata():
    run = EvaluationRun(results=[])

    experiment = Experiment(
        name="baseline-v1",
        model="gpt-model",
        model_version="v2",
        prompt="Answer the question.",
        prompt_version="prompt-v3",
        dataset="math",
        dataset_version="dataset-v5",
        run=run,
    )

    data = experiment.to_dict()

    assert data["model_version"] == "v2"
    assert data["prompt_version"] == "prompt-v3"
    assert data["dataset_version"] == "dataset-v5"

def test_compare_experiments_detects_model_version_change():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="gpt-model",
        model_version="v1",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    current = Experiment(
        name="current-v2",
        model="gpt-model",
        model_version="v2",
        prompt="Answer the question.",
        dataset="math-v1",
        run=run,
    )

    comparison = compare_experiments(baseline, current)

    assert comparison.model_version_changed is True


def test_compare_experiments_detects_prompt_version_change():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="gpt-model",
        prompt="Answer the question.",
        prompt_version="v1",
        dataset="math-v1",
        run=run,
    )

    current = Experiment(
        name="current-v2",
        model="gpt-model",
        prompt="Answer the question.",
        prompt_version="v2",
        dataset="math-v1",
        run=run,
    )

    comparison = compare_experiments(baseline, current)

    assert comparison.prompt_version_changed is True


def test_compare_experiments_detects_dataset_version_change():
    run = EvaluationRun(results=[])

    baseline = Experiment(
        name="baseline-v1",
        model="gpt-model",
        prompt="Answer the question.",
        dataset="math",
        dataset_version="v1",
        run=run,
    )

    current = Experiment(
        name="current-v2",
        model="gpt-model",
        prompt="Answer the question.",
        dataset="math",
        dataset_version="v2",
        run=run,
    )

    comparison = compare_experiments(baseline, current)

    assert comparison.dataset_version_changed is True