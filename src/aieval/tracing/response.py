from dataclasses import dataclass

from aieval.tracing.usage import ModelUsage


@dataclass
class ModelResponse:
    output: str
    usage: ModelUsage | None = None
    finish_reason: str | None = None
    response_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelResponse":
        usage_data = data.get("usage")

        return cls(
            output=data["output"],
            usage=(
                ModelUsage(
                    input_tokens=usage_data["input_tokens"],
                    output_tokens=usage_data["output_tokens"],
                    input_cost=usage_data["input_cost"],
                    output_cost=usage_data["output_cost"],
                )
                if usage_data is not None
                else None
            ),
            finish_reason=data.get("finish_reason"),
            response_id=data.get("response_id"),
        )