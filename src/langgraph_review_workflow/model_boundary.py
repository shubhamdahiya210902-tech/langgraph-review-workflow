"""Model boundary configuration.

The demo runs deterministically by default. This config documents the intended
swap point for an OpenAI-compatible LiteLLM proxy or hosted chat model.
"""

from typing import Literal

from pydantic import BaseModel, Field


ModelProvider = Literal["deterministic", "litellm-compatible"]


class ModelBoundaryConfig(BaseModel):
    """Configuration for model-backed structured-output steps."""

    provider: ModelProvider = "deterministic"
    model: str = "deterministic-local"
    api_base: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL, such as a LiteLLM proxy.",
    )
    temperature: float = 0.0
    structured_outputs_required: bool = True


DEFAULT_MODEL_BOUNDARY = ModelBoundaryConfig()
