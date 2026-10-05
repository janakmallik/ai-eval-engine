from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResponse:
    output: Any
    result_count: int | None = None
