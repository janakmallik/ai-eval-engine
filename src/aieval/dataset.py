from dataclasses import dataclass


@dataclass
class EvalCase:
    id: str
    input: str
    expected: str
