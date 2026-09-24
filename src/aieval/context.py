from dataclasses import dataclass

from aieval.dataset import EvalCase


@dataclass
class EvaluationContext:
    case: EvalCase
    actual: str
