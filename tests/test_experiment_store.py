# ExperimentStore
# We'll start with a simple JSON-backed experiment store. This is much more
# useful for the project than an in-memory store because experiments need to
# survive process restarts and become reproducible artifacts.

import json

import pytest

from aieval.experiment import Experiment
from aieval.run import EvaluationRun
from aieval.store import ExperimentStore


def create_experiment(name="baseline"):
    return Experiment(
        name=name,
        model="qa-model",
        model_version="v1",
        prompt="Answer using the context.",
        prompt_version="v1",
        dataset="qa-dataset",
        dataset_version="v1",
        run=EvaluationRun(results=[]),
        metadata={"environment": "test"},
        evaluator_config={
            "exact_match": {
                "case_sensitive": False,
            }
        },
    )


def test_experiment_store_can_be_created(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    assert store.path == tmp_path / "experiments.json"


def test_experiment_store_saves_experiment(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    assert store.path.exists()


def test_experiment_store_file_contains_json(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    data = json.loads(store.path.read_text())

    assert isinstance(data, dict)


def test_experiment_store_saved_experiment_has_experiment_id(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    data = json.loads(store.path.read_text())

    assert experiment.experiment_id in data


def test_experiment_store_can_load_experiment(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert isinstance(loaded, Experiment)


def test_experiment_store_preserves_experiment_identity(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.experiment_id == experiment.experiment_id


def test_experiment_store_preserves_experiment_name(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment(name="production")

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.name == "production"


def test_experiment_store_preserves_experiment_definition(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.model == "qa-model"
    assert loaded.model_version == "v1"
    assert loaded.prompt == "Answer using the context."
    assert loaded.prompt_version == "v1"
    assert loaded.dataset == "qa-dataset"
    assert loaded.dataset_version == "v1"


def test_experiment_store_preserves_metadata(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.metadata == {"environment": "test"}


def test_experiment_store_preserves_evaluator_config(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.evaluator_config == {
        "exact_match": {
            "case_sensitive": False,
        }
    }


def test_experiment_store_rejects_unknown_experiment(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    with pytest.raises(KeyError, match="unknown"):
        store.load("unknown")


def test_experiment_store_can_save_multiple_experiments(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    first = create_experiment(name="baseline")

    second = create_experiment(name="candidate")
    second.model_version = "v2"

    store.save(first)
    store.save(second)

    loaded_first = store.load(first.experiment_id)
    loaded_second = store.load(second.experiment_id)

    assert loaded_first.name == "baseline"
    assert loaded_second.name == "candidate"


def test_experiment_store_overwrites_same_experiment(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    experiment = create_experiment(name="baseline")
    store.save(experiment)

    experiment.name = "updated"
    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.name == "updated"


def test_experiment_store_does_not_mutate_experiment_on_save(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")
    experiment = create_experiment()

    original_id = experiment.experiment_id

    store.save(experiment)

    assert experiment.experiment_id == original_id


def test_experiment_store_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "experiments.json"
    store = ExperimentStore(path)

    store.save(create_experiment())

    assert path.exists()
