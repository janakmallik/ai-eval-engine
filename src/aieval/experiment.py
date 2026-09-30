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