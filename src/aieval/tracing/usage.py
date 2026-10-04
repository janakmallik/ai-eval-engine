from dataclasses import dataclass


@dataclass
class ModelUsage:
    input_tokens: int
    output_tokens: int
    input_cost: float = 0.0
    output_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")

        if self.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")

        if self.input_cost < 0:
            raise ValueError("input_cost cannot be negative")

        if self.output_cost < 0:
            raise ValueError("output_cost cannot be negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }
