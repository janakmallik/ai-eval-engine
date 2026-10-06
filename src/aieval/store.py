import json
from pathlib import Path

from aieval.experiment import Experiment
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


class ExperimentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, experiment: Experiment) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = self._read()

        data[experiment.experiment_id] = experiment.to_dict()

        self.path.write_text(
            json.dumps(
                data,
                indent=2,
                sort_keys=True,
            )
        )

    def load(self, experiment_id: str) -> Experiment:
        data = self._read()

        if experiment_id not in data:
            raise KeyError(experiment_id)

        return self._from_dict(data[experiment_id])

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}

        return json.loads(self.path.read_text())

    @staticmethod
    def _from_dict(data: dict) -> Experiment:
        run_data = data["run"]

        results = [
            EvaluationResult(
                case_id=result["case_id"],
                evaluator_name=result["evaluator_name"],
                expected=result["expected"],
                actual=result["actual"],
                score=result["score"],
                passed=result["passed"],
            )
            for result in run_data["results"]
        ]

        run = EvaluationRun(
            results=results,
            metadata=run_data.get("metadata", {}),
            trace=None,
        )

        return Experiment(
            name=data["name"],
            model=data["model"],
            model_version=data.get("model_version"),
            prompt=data.get("prompt", ""),
            prompt_version=data.get("prompt_version"),
            dataset=data["dataset"],
            dataset_version=data.get("dataset_version"),
            run=run,
            metadata=data.get("metadata", {}),
            evaluator_config=data.get("evaluator_config", {}),
        )
