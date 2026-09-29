import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    input: str
    expected: str


@dataclass
class EvalDataset:
    cases: list[EvalCase]

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalDataset":
        input_path = Path(path)

        data = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        if not isinstance(data, list):
            raise ValueError("Dataset JSON must contain a list of cases")

        cases = []

        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each dataset case must be an object")

            required_fields = {"id", "input", "expected"}

            if not required_fields.issubset(item):
                raise ValueError(
                    "Each dataset case must contain: id, input, expected"
                )

            cases.append(EvalCase(**item))

        return cls(cases=cases)

    def to_json(self, path: str | Path) -> None:
        output_path = Path(path)

        data = [
            {
                "id": case.id,
                "input": case.input,
                "expected": case.expected,
            }
            for case in self.cases
        ]

        output_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def __iter__(self):
        return iter(self.cases)