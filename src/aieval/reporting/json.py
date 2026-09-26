import json
from pathlib import Path

from aieval.run import EvaluationRun


class JsonReporter:
    def render(self, run: EvaluationRun) -> str:
        return json.dumps(run.to_dict(), indent=2)

    def write(self, run: EvaluationRun, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(self.render(run), encoding="utf-8")