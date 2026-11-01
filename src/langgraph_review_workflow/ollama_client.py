"""Minimal Ollama HTTP client for structured chat and embeddings.

Uses only the standard library so the default install stays dependency-light.
Structured outputs use Ollama's `format` field with a Pydantic JSON schema, and
every response is validated against that schema before it reaches the graph.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from langgraph_review_workflow.model_boundary import (
    DEFAULT_OLLAMA_BASE_URL,
    ModelBoundaryConfig,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class StructuredChatClient(Protocol):
    """Interface used by live steps, so tests can inject a fake client."""

    def structured_chat(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        temperature: float,
        seed: int | None = None,
    ) -> SchemaT: ...


class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_boundary(cls, boundary: ModelBoundaryConfig) -> OllamaClient:
        return cls(
            model=boundary.model,
            base_url=boundary.api_base or DEFAULT_OLLAMA_BASE_URL,
            timeout_seconds=boundary.timeout_seconds,
        )

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama {path} returned {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise OllamaError(f"Ollama is not reachable at {self.base_url}: {error}") from error

    def is_available(self) -> bool:
        try:
            self._request("/api/version")
        except OllamaError:
            return False
        return True

    def installed_models(self) -> list[str]:
        tags = self._request("/api/tags")
        return [str(model["name"]) for model in tags.get("models", [])]

    def running_models(self) -> list[dict[str, Any]]:
        """Loaded models with total size and the share resident in GPU memory."""

        return list(self._request("/api/ps").get("models", []))

    def structured_chat(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        temperature: float,
        seed: int | None = None,
    ) -> SchemaT:
        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed

        response = self._request(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": schema.model_json_schema(),
                "stream": False,
                "options": options,
            },
        )
        content = response.get("message", {}).get("content", "")
        return schema.model_validate_json(content)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._request("/api/embed", {"model": self.model, "input": texts})
        return [list(map(float, vector)) for vector in response.get("embeddings", [])]
