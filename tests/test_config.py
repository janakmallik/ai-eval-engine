import json

import pytest

from aieval.config import EvaluationConfig
from aieval.evaluators.exact_match import ExactMatchEvaluator


def test_evaluation_config_stores_evaluation_definition():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.model == "qa-model"
    assert config.dataset == "qa-dataset"


def test_evaluation_config_stores_evaluators():
    evaluator = ExactMatchEvaluator()

    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        evaluators=[evaluator],
    )

    assert config.evaluators == [evaluator]


def test_evaluation_config_stores_model_version():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        model_version="v2",
    )

    assert config.model_version == "v2"


def test_evaluation_config_stores_prompt():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        prompt="Answer the question using the provided context.",
    )

    assert config.prompt == "Answer the question using the provided context."


def test_evaluation_config_stores_prompt_version():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        prompt="Answer the question using the provided context.",
        prompt_version="v3",
    )

    assert config.prompt_version == "v3"




def test_evaluation_config_stores_dataset_version():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        dataset_version="v5",
    )

    assert config.dataset_version == "v5"


def test_evaluation_config_defaults_optional_versions_to_none():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.model_version is None
    assert config.prompt is None
    assert config.prompt_version is None
    assert config.dataset_version is None


def test_evaluation_config_defaults_evaluators_to_empty():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert list(config.evaluators) == []


def test_evaluation_config_stores_metadata():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        metadata={"environment": "test"},
    )

    assert config.metadata == {"environment": "test"}




def test_evaluation_config_copies_evaluators():
    evaluators = [ExactMatchEvaluator()]

    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        evaluators=evaluators,
    )

    evaluators.append(ExactMatchEvaluator())

    assert len(config.evaluators) == 1


def test_evaluation_config_stores_evaluator_config():
    evaluator_config = {
        "exact_match": {
            "case_sensitive": False,
        }
    }

    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        evaluator_config=evaluator_config,
    )

    assert config.evaluator_config == evaluator_config


def test_evaluation_config_defaults_evaluator_config_to_empty():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.evaluator_config == {}


def test_evaluation_config_stores_model_provider():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        model_provider="openai",
    )

    assert config.model_provider == "openai"




def test_evaluation_config_stores_retrieval_settings():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        retrieval_top_k=10,
    )

    assert config.retrieval_top_k == 10




def test_evaluation_config_stores_tracing_setting():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        enable_tracing=True,
    )

    assert config.enable_tracing is True




def test_evaluation_config_stores_metadata_and_versions_together():
    config = EvaluationConfig(
        model="qa-model",
        model_version="v2",
        dataset="qa-dataset",
        dataset_version="v5",
        prompt="Answer using the context.",
        prompt_version="v3",
        metadata={"environment": "production"},
    )

    assert config.model == "qa-model"
    assert config.model_version == "v2"
    assert config.dataset == "qa-dataset"
    assert config.dataset_version == "v5"
    assert config.prompt == "Answer using the context."
    assert config.prompt_version == "v3"
    assert config.metadata == {"environment": "production"}


def test_evaluation_config_has_stable_config_id():
    config = EvaluationConfig(
        model="qa-model",
        model_version="v2",
        dataset="qa-dataset",
        dataset_version="v5",
        prompt="Answer using the context.",
        prompt_version="v3",
    )

    assert isinstance(config.config_id, str)
    assert len(config.config_id) == 64






def test_evaluation_config_id_changes_when_model_version_changes():
    config1 = EvaluationConfig(
        model="qa-model",
        model_version="v1",
        dataset="qa-dataset",
    )

    config2 = EvaluationConfig(
        model="qa-model",
        model_version="v2",
        dataset="qa-dataset",
    )

    assert config1.config_id != config2.config_id




def test_evaluation_config_to_dict_contains_config_id():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    data = config.to_dict()

    assert data["config_id"] == config.config_id


def test_evaluation_config_to_dict_contains_core_fields():
    config = EvaluationConfig(
        model="qa-model",
        model_version="v2",
        dataset="qa-dataset",
        dataset_version="v5",
        prompt="Answer using the context.",
        prompt_version="v3",
    )

    data = config.to_dict()

    assert data["model"] == "qa-model"
    assert data["model_version"] == "v2"
    assert data["dataset"] == "qa-dataset"
    assert data["dataset_version"] == "v5"
    assert data["prompt"] == "Answer using the context."
    assert data["prompt_version"] == "v3"


def test_evaluation_config_to_dict_contains_execution_settings():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        model_provider="openai",
        retrieval_top_k=10,
        enable_tracing=True,
    )

    data = config.to_dict()

    assert data["model_provider"] == "openai"
    assert data["retrieval_top_k"] == 10
    assert data["enable_tracing"] is True


def test_evaluation_config_to_dict_contains_metadata():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        metadata={"environment": "test"},
    )

    data = config.to_dict()

    assert data["metadata"] == {"environment": "test"}


def test_evaluation_config_to_dict_is_json_serializable():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        metadata={"environment": "test"},
    )

    data = config.to_dict()

    json.dumps(data)


def test_evaluation_config_requires_model():
    with pytest.raises(TypeError):
        EvaluationConfig(
            dataset="qa-dataset",
        )


def test_evaluation_config_requires_dataset():
    with pytest.raises(TypeError):
        EvaluationConfig(
            model="qa-model",
        )






def test_evaluation_config_rejects_invalid_retrieval_top_k():
    with pytest.raises(ValueError, match="retrieval_top_k"):
        EvaluationConfig(
            model="qa-model",
            dataset="qa-dataset",
            retrieval_top_k=0,
        )


def test_evaluation_config_accepts_positive_retrieval_top_k():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        retrieval_top_k=10,
    )

    assert config.retrieval_top_k == 10


def test_evaluation_config_defaults_retrieval_top_k():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.retrieval_top_k == 5


def test_evaluation_config_defaults_model_provider():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.model_provider == "unknown"


def test_evaluation_config_defaults_tracing_to_disabled():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.enable_tracing is False


def test_evaluation_config_defaults_metadata_to_empty_dict():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.metadata == {}


def test_evaluation_config_defaults_evaluator_config_to_empty_dict():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.evaluator_config == {}






def test_evaluation_config_id_changes_when_retrieval_top_k_changes():
    config1 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        retrieval_top_k=5,
    )

    config2 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        retrieval_top_k=10,
    )

    assert config1.config_id != config2.config_id


def test_evaluation_config_id_changes_when_tracing_changes():
    config1 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        enable_tracing=False,
    )

    config2 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        enable_tracing=True,
    )

    assert config1.config_id != config2.config_id




def test_evaluation_config_id_does_not_depend_on_metadata():
    config1 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        metadata={"environment": "test"},
    )

    config2 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        metadata={"environment": "production"},
    )

    assert config1.config_id == config2.config_id


def test_evaluation_config_copies_metadata():
    metadata = {"team": "research"}

    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        metadata=metadata,
    )

    metadata["team"] = "production"

    assert config.metadata["team"] == "research"


def test_evaluation_config_copies_evaluator_config():
    evaluator_config = {
        "exact_match": {
            "case_sensitive": False,
        }
    }

    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        evaluator_config=evaluator_config,
    )

    evaluator_config["exact_match"]["case_sensitive"] = True

    assert config.evaluator_config["exact_match"]["case_sensitive"] is False


def test_evaluation_config_default_values():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    assert config.model_version is None
    assert config.prompt is None
    assert config.prompt_version is None
    assert config.dataset_version is None
    assert config.model_provider == "unknown"
    assert config.retrieval_top_k == 5
    assert config.enable_tracing is False
    assert config.evaluators == []
    assert config.metadata == {}
    assert config.evaluator_config == {}


def test_evaluation_config_rejects_empty_model():
    with pytest.raises(ValueError, match="model must not be empty"):
        EvaluationConfig(
            model="",
            dataset="qa-dataset",
        )


def test_evaluation_config_rejects_empty_dataset():
    with pytest.raises(ValueError, match="dataset must not be empty"):
        EvaluationConfig(
            model="qa-model",
            dataset="",
        )


def test_evaluation_config_rejects_zero_retrieval_top_k():
    with pytest.raises(
        ValueError,
        match="retrieval_top_k must be greater than 0",
    ):
        EvaluationConfig(
            model="qa-model",
            dataset="qa-dataset",
            retrieval_top_k=0,
        )


def test_evaluation_config_rejects_negative_retrieval_top_k():
    with pytest.raises(
        ValueError,
        match="retrieval_top_k must be greater than 0",
    ):
        EvaluationConfig(
            model="qa-model",
            dataset="qa-dataset",
            retrieval_top_k=-1,
        )


def test_evaluation_config_id_is_deterministic():
    config1 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        model_version="v1",
        prompt="Answer the question.",
    )

    config2 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        model_version="v1",
        prompt="Answer the question.",
    )

    assert config1.config_id == config2.config_id


def test_evaluation_config_id_changes_when_model_changes():
    config1 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    config2 = EvaluationConfig(
        model="different-model",
        dataset="qa-dataset",
    )

    assert config1.config_id != config2.config_id


def test_evaluation_config_id_changes_when_prompt_changes():
    config1 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        prompt="Answer the question.",
    )

    config2 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        prompt="Answer briefly.",
    )

    assert config1.config_id != config2.config_id


def test_evaluation_config_id_changes_when_evaluator_config_changes():
    config1 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        evaluator_config={
            "exact_match": {
                "case_sensitive": False,
            }
        },
    )

    config2 = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        evaluator_config={
            "exact_match": {
                "case_sensitive": True,
            }
        },
    )

    assert config1.config_id != config2.config_id


def test_evaluation_config_to_dict_includes_config_id():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
    )

    data = config.to_dict()

    assert data["config_id"] == config.config_id


def test_evaluation_config_to_dict_includes_all_configuration_fields():
    config = EvaluationConfig(
        model="qa-model",
        dataset="qa-dataset",
        model_version="v2",
        prompt="Answer the question.",
        prompt_version="v3",
        dataset_version="v5",
        model_provider="openai",
        retrieval_top_k=10,
        enable_tracing=True,
        metadata={"team": "research"},
        evaluator_config={
            "exact_match": {
                "case_sensitive": False,
            }
        },
    )

    data = config.to_dict()

    assert data == {
        "config_id": config.config_id,
        "model": "qa-model",
        "model_version": "v2",
        "dataset": "qa-dataset",
        "dataset_version": "v5",
        "prompt": "Answer the question.",
        "prompt_version": "v3",
        "model_provider": "openai",
        "retrieval_top_k": 10,
        "enable_tracing": True,
        "metadata": {"team": "research"},
        "evaluator_config": {
            "exact_match": {
                "case_sensitive": False,
            }
        },
    }
