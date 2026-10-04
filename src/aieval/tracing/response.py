from dataclasses import dataclass

from aieval.tracing.usage import ModelUsage


@dataclass
class ModelResponse:
    output: str
    usage: ModelUsage | None = None
