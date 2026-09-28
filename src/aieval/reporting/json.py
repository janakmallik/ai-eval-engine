import json
from pathlib import Path

from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


class JsonReporter:
    def render(self, run: EvaluationRun) -> str:
        return json.dumps(run.to_dict(), indent=2)

    def write(self, run: EvaluationRun, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(self.render(run), encoding="utf-8")

    def read(self, path: str | Path) -> EvaluationRun:
        input_path = Path(path)
        data = json.loads(input_path.read_text(encoding="utf-8"))

        results = [
            EvaluationResult(**result)
            for result in data["results"]
        ]

        return EvaluationRun(
            results=results,
            metadata=data.get("metadata", {}),
        )