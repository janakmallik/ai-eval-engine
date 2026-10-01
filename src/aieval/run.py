import hashlib
import json

from dataclasses import dataclass, field

from aieval.result import EvaluationResult
from aieval.summary import EvaluationSummary
from aieval.tracing.trace import Trace

@dataclass
class EvaluationRun:
    results: list[EvaluationResult]
    metadata: dict[str, object] = field(default_factory=dict)
    trace: Trace | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def score(self) -> float:
        if self.total == 0:
            return 0.0

        return sum(result.score for result in self.results) / self.total

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0

        return self.passed / self.total

    @property
    def run_id(self) -> str:
        payload = {
            "results": [result.to_dict() for result in self.results],
            "metadata": self.metadata,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def by_evaluator(self, evaluator_name: str) -> list[EvaluationResult]:
        return [
            result
            for result in self.results
            if result.evaluator_name == evaluator_name
        ]

    def summary(self, evaluator_name: str) -> EvaluationSummary:
        results = self.by_evaluator(evaluator_name)

        total = len(results)
        passed = sum(result.passed for result in results)
        failed = total - passed

        if total == 0:
            score = 0.0
            pass_rate = 0.0
        else:
            score = sum(result.score for result in results) / total
            pass_rate = passed / total

        return EvaluationSummary(
            evaluator_name=evaluator_name,
            total=total,
            passed=passed,
            failed=failed,
            score=score,
            pass_rate=pass_rate,
        )

    def summaries(self) -> dict[str, EvaluationSummary]:
        evaluator_names = {result.evaluator_name for result in self.results}

        return {
            evaluator_name: self.summary(evaluator_name)
            for evaluator_name in evaluator_names
        }

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "schema_version": 1,
            "results": [result.to_dict() for result in self.results],
            "metadata": self.metadata,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "score": self.score,
            "pass_rate": self.pass_rate,
            "summaries": {
                evaluator_name: summary.to_dict()
                for evaluator_name, summary in self.summaries().items()
            },
        }
