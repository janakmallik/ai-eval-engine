from aieval import (
    EvaluationConfig,
    EvaluationRun,
    Experiment,
    ExperimentStore,
)


def test_experiment_public_api(tmp_path):
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    run = EvaluationRun(results=[])

    experiment = Experiment.from_config(
        name="baseline",
        config=config,
        run=run,
    )

    store = ExperimentStore(tmp_path / "experiments.json")

    store.save(experiment)
    loaded = store.load(experiment.experiment_id)

    assert config.model == "qa-model"
    assert experiment.name == "baseline"
    assert experiment.run is run
    assert loaded.name == "baseline"
    assert loaded.experiment_id == experiment.experiment_id
