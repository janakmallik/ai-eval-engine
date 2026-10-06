import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from aieval.evaluators.base import Evaluator


@dataclass
class EvaluationConfig:
    model: str
    dataset: str
    evaluators: Iterable[Evaluator] = field(default_factory=list)

    model_version: str | None = None

    prompt: str | None = None
    prompt_version: str | None = None

    dataset_version: str | None = None

    model_provider: str = "unknown"
    retrieval_top_k: int = 5
    enable_tracing: bool = False

    metadata: dict[str, object] = field(default_factory=dict)

    evaluator_config: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def config_id(self) -> str:
        data = {
            "model": self.model,
            "model_version": self.model_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "prompt": self.prompt,
            "prompt_version": self.prompt_version,
            "model_provider": self.model_provider,
            "retrieval_top_k": self.retrieval_top_k,
            "enable_tracing": self.enable_tracing,
            "evaluator_config": self.evaluator_config,
        }

        serialized = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(serialized.encode()).hexdigest()

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")

        if not self.dataset:
            raise ValueError("dataset must not be empty")

        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than 0")

        self.evaluators = list(self.evaluators)
        self.metadata = dict(self.metadata)
        self.evaluator_config = {
            name: dict(settings) for name, settings in self.evaluator_config.items()
        }

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "model": self.model,
            "model_version": self.model_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "prompt": self.prompt,
            "prompt_version": self.prompt_version,
            "model_provider": self.model_provider,
            "retrieval_top_k": self.retrieval_top_k,
            "enable_tracing": self.enable_tracing,
            "metadata": self.metadata,
            "evaluator_config": self.evaluator_config,
        }
