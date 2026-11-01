"""Model boundary configuration.

The demo runs deterministically by default. The `ollama` provider swaps the
groundedness and drafting steps for structured-output calls to a local Ollama
model; the graph, state, and Pydantic schemas stay the same. The
`litellm-compatible` provider documents the intended swap point for an
OpenAI-compatible LiteLLM proxy or hosted chat model.
"""

from typing import Literal

from pydantic import BaseModel, Field


ModelProvider = Literal["deterministic", "litellm-compatible", "ollama"]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
# A different model family from the drafter, so drafts are not graded by their own model.
DEFAULT_OLLAMA_JUDGE_MODEL = "llama3.2:3b"


class ModelBoundaryConfig(BaseModel):
    """Configuration for model-backed structured-output steps."""

    provider: ModelProvider = "deterministic"
    model: str = "deterministic-local"
    api_base: str | None = Field(
        default=None,
        description="Optional base URL, such as a LiteLLM proxy or a local Ollama server.",
    )
    temperature: float = 0.0
    seed: int | None = Field(
        default=None,
        description="Optional sampling seed; leave unset for independent live runs.",
    )
    timeout_seconds: float = 300.0
    structured_outputs_required: bool = True

    @property
    def is_live(self) -> bool:
        """Whether model-backed steps should call a live model."""

        return self.provider == "ollama"


DEFAULT_MODEL_BOUNDARY = ModelBoundaryConfig()


def ollama_model_boundary(
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.7,
    api_base: str = DEFAULT_OLLAMA_BASE_URL,
    seed: int | None = None,
) -> ModelBoundaryConfig:
    """Build a model boundary that routes model-backed steps to local Ollama."""

    return ModelBoundaryConfig(
        provider="ollama",
        model=model,
        api_base=api_base,
        temperature=temperature,
        seed=seed,
    )
