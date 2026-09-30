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
    metadata: dict[str, object] = field(default_factory=dict)
    evaluator_config: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
        data = {
            "model": self.model,
            "prompt": self.prompt,
            "dataset": self.dataset,
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
            "prompt": self.prompt,
            "dataset": self.dataset,
            "run": self.run.to_dict(),
            "metadata": self.metadata,
            "evaluator_config": self.evaluator_config,
        }