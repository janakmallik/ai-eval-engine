import hashlib
import json
from dataclasses import dataclass, field

from aieval.run import EvaluationRun


@dataclass
class Experiment:
    name: str
    model: str
    prompt: str
    dataset: str
    run: EvaluationRun
    model_version: str | None = None
    prompt_version: str | None = None
    dataset_version: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    evaluator_config: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "model_version": self.model_version,
            "prompt": self.prompt,
            "prompt_version": self.prompt_version,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "run": self.run.to_dict(),
            "metadata": self.metadata,
            "evaluator_config": self.evaluator_config,
        }
