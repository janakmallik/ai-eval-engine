from aieval import EvaluationConfig, EvaluationRun, Experiment, ExperimentStore


def test_experiment_can_be_created_from_public_api():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-v1",
        model_version="v1",
    )

    run = EvaluationRun(results=[])

    experiment = Experiment.from_config(
        name="baseline",
        config=config,
        run=run,
    )

    assert experiment.name == "baseline"
    assert experiment.model == "qa-model"
    assert experiment.dataset == "qa-v1"
    assert experiment.model_version == "v1"

    def test_experiment_store_is_available_from_public_api():
        assert ExperimentStore is not None
