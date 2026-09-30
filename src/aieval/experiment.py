from dataclasses import dataclass

from aieval.run import EvaluationRun


@dataclass
class Experiment:
    name: str
    model: str
    prompt: str
    dataset: str
    run: EvaluationRun

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "prompt": self.prompt,
            "dataset": self.dataset,
            "run": self.run.to_dict(),
        }