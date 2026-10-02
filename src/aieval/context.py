from dataclasses import dataclass, field

from aieval.dataset import EvalCase


@dataclass(frozen=True)
class EvaluationContext:
    case: EvalCase
    actual: str
    metadata: dict[str, object] = field(default_factory=dict)
