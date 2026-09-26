import json

from aieval.run import EvaluationRun


class JsonReporter:
    def render(self, run: EvaluationRun) -> str:
        return json.dumps(run.to_dict(), indent=2)
    