# ExperimentStore
# We'll start with a simple JSON-backed experiment store. This is much more
# useful for the project than an in-memory store because experiments need to
# survive process restarts and become reproducible artifacts.

import json

import pytest

from aieval.config import EvaluationConfig
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


def test_experiment_store_can_list_experiments(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    first = create_experiment(name="baseline")
    second = create_experiment(name="candidate")
    second.model_version = "v2"

    store.save(first)
    store.save(second)

    experiments = store.list()

    assert len(experiments) == 2
    assert {experiment.name for experiment in experiments} == {
        "baseline",
        "candidate",
    }


def test_experiment_store_list_returns_empty_when_store_does_not_exist(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    assert store.list() == []


def test_experiment_store_list_returns_experiments_with_their_runs(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    experiment = create_experiment(name="baseline")

    store.save(experiment)

    experiments = store.list()

    assert len(experiments) == 1
    assert experiments[0].name == "baseline"
    assert experiments[0].run.to_dict() == experiment.run.to_dict()


def test_experiment_store_list_does_not_modify_store(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    first = create_experiment(name="baseline")
    second = create_experiment(name="candidate")
    second.model_version = "v2"

    store.save(first)
    store.save(second)

    before = store.path.read_text()

    store.list()

    after = store.path.read_text()

    assert after == before


def test_experiment_store_can_find_by_name(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    baseline = create_experiment(name="baseline")
    candidate = create_experiment(name="candidate")

    store.save(baseline)
    store.save(candidate)

    result = store.find(name="candidate")

    assert result is not None
    assert result.name == "candidate"


def test_experiment_store_find_returns_none_when_name_not_found(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    store.save(create_experiment(name="baseline"))

    result = store.find(name="missing")

    assert result is None


def test_experiment_store_can_find_by_model_and_version(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    baseline = create_experiment(name="baseline")
    candidate = create_experiment(name="candidate")
    candidate.model = "qa-model-v2"
    candidate.model_version = "v2"

    store.save(baseline)
    store.save(candidate)

    result = store.find(
        model="qa-model-v2",
        model_version="v2",
    )

    assert result is not None
    assert result.name == "candidate"


def test_experiment_store_find_can_combine_filters(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    first = create_experiment(name="baseline")
    first.model_version = "v1"

    second = create_experiment(name="candidate")
    second.model_version = "v2"

    store.save(first)
    store.save(second)

    result = store.find(
        name="candidate",
        model_version="v2",
    )

    assert result is not None
    assert result.name == "candidate"


def test_experiment_store_find_does_not_modify_store(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    experiment = create_experiment(name="baseline")
    store.save(experiment)

    before = store.path.read_text()

    store.find(name="baseline")

    after = store.path.read_text()

    assert after == before


def test_experiment_store_can_delete_experiment(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    experiment = create_experiment(name="baseline")
    store.save(experiment)

    deleted = store.delete(experiment.experiment_id)

    assert deleted is True
    with pytest.raises(KeyError):
        store.load(experiment.experiment_id)


def test_experiment_store_delete_returns_false_when_experiment_does_not_exist(
    tmp_path,
):
    store = ExperimentStore(tmp_path / "experiments.json")

    deleted = store.delete("missing")

    assert deleted is False


def test_experiment_store_delete_does_not_delete_other_experiments(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    first = create_experiment(name="baseline")
    second = create_experiment(name="candidate")
    second.model_version = "v2"

    store.save(first)
    store.save(second)

    deleted = store.delete(first.experiment_id)

    assert deleted is True
    with pytest.raises(KeyError):
        store.load(first.experiment_id)
    assert store.load(second.experiment_id).name == "candidate"


def test_experiment_store_delete_updates_persisted_store(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    first = create_experiment(name="baseline")
    second = create_experiment(name="candidate")
    second.model_version = "v2"

    store.save(first)
    store.save(second)

    store.delete(first.experiment_id)

    data = json.loads(store.path.read_text())

    assert first.experiment_id not in data
    assert second.experiment_id in data


def test_experiment_store_delete_does_not_modify_store_when_missing(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.json")

    experiment = create_experiment(name="baseline")
    store.save(experiment)

    before = store.path.read_text()

    deleted = store.delete("missing")

    after = store.path.read_text()

    assert deleted is False
    assert after == before


def test_experiment_store_preserves_trace(tmp_path):
    from aieval.tracing.trace import Trace

    store = ExperimentStore(tmp_path / "experiments.json")

    trace = Trace()
    with trace.start_span("evaluation"):
        pass

    experiment = create_experiment()
    experiment.run.trace = trace

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.run.trace is not None


def test_experiment_store_preserves_trace_data(tmp_path):
    from aieval.tracing.trace import Trace

    store = ExperimentStore(tmp_path / "experiments.json")

    trace = Trace()

    with trace.start_span("evaluation") as span:
        span.set_attribute("test.key", "test-value")

    trace.end()

    experiment = create_experiment()
    experiment.run.trace = trace

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.run.trace is not None
    assert loaded.run.trace.trace_id == trace.trace_id
    assert len(loaded.run.trace.spans) == 1
    assert loaded.run.trace.spans[0].name == "evaluation"
    assert loaded.run.trace.spans[0].attributes["test.key"] == "test-value"


def test_experiment_store_preserves_config_based_experiment_id(tmp_path):
    config = EvaluationConfig(
        model="qa-model",
        model_version="v1",
        dataset="qa-dataset",
        dataset_version="v1",
        prompt="Answer using the context.",
        prompt_version="v1",
    )

    experiment = Experiment.from_config(
        name="baseline",
        config=config,
        run=EvaluationRun(results=[]),
    )

    store = ExperimentStore(tmp_path / "experiments.json")

    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.config is not None
    assert loaded.config.config_id == config.config_id
    assert loaded.experiment_id == experiment.experiment_id


def test_experiment_store_overwrites_config_based_experiment(tmp_path):
    config = EvaluationConfig(
        model="qa-model",
        model_version="v1",
        dataset="qa-dataset",
        dataset_version="v1",
        prompt="Answer using the context.",
        prompt_version="v1",
    )

    experiment = Experiment.from_config(
        name="baseline",
        config=config,
        run=EvaluationRun(results=[]),
    )

    store = ExperimentStore(tmp_path / "experiments.json")

    store.save(experiment)

    experiment.name = "updated"
    store.save(experiment)

    loaded = store.load(experiment.experiment_id)

    assert loaded.name == "updated"
    assert loaded.experiment_id == experiment.experiment_id


def test_experiment_store_can_find_config_based_experiment(tmp_path):
    config = EvaluationConfig(
        model="qa-model",
        model_version="v1",
        dataset="qa-dataset",
        dataset_version="v1",
        prompt="Answer using the context.",
        prompt_version="v1",
    )

    experiment = Experiment.from_config(
        name="baseline",
        config=config,
        run=EvaluationRun(results=[]),
    )

    store = ExperimentStore(tmp_path / "experiments.json")
    store.save(experiment)

    found = store.find(
        name="baseline",
        model="qa-model",
        model_version="v1",
    )

    assert found is not None
    assert found.experiment_id == experiment.experiment_id
