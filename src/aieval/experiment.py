import hashlib
import json
from dataclasses import dataclass, field

from aieval.config import EvaluationConfig
from aieval.run import EvaluationRun


@dataclass
class Experiment:
    name: str

    # Legacy fields are kept for backward compatibility.
    model: str = ""
    prompt: str = ""
    dataset: str = ""
    run: EvaluationRun = field(default_factory=lambda: EvaluationRun(results=[]))

    model_version: str | None = None
    prompt_version: str | None = None
    dataset_version: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    evaluator_config: dict[str, dict[str, object]] = field(default_factory=dict)

    # New configuration source of truth.
    config: EvaluationConfig | None = None

    def __post_init__(self) -> None:
        if self.config is not None:
            self.model = self.config.model
            self.prompt = self.config.prompt or ""
            self.dataset = self.config.dataset

            self.model_version = self.config.model_version
            self.prompt_version = self.config.prompt_version
            self.dataset_version = self.config.dataset_version

            self.metadata = dict(self.config.metadata)
            self.evaluator_config = {
                name: dict(settings)
                for name, settings in self.config.evaluator_config.items()
            }

    @property
    def experiment_id(self) -> str:
        if self.config is not None:
            return self.config.config_id

        data = {
            "model": self.model,
            "model_version": self.model_version,
            "prompt": self.prompt,
            "prompt_version": self.prompt_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "evaluator_config": self.evaluator_config,
        }

        serialized = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(serialized.encode()).hexdigest()

    @classmethod
    def from_config(
        cls,
        name: str,
        config: EvaluationConfig,
        run: EvaluationRun,
    ) -> "Experiment":
        return cls(
            name=name,
            config=config,
            run=run,
        )

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "model": self.model,
            "model_version": self.model_version,
            "prompt": self.prompt,
            "prompt_version": self.prompt_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "run": self.run.to_dict(),
            "metadata": dict(self.metadata),
            "evaluator_config": {
                name: dict(settings) for name, settings in self.evaluator_config.items()
            },
        }

        if self.config is not None:
            data["config"] = self.config.to_dict()

        return data
