import json
import os
from pathlib import Path

import pytest

from langgraph_review_workflow.model_boundary import DEFAULT_OLLAMA_BASE_URL
from langgraph_review_workflow.ollama_client import OllamaClient
from langgraph_review_workflow.steps.evidence import map_evidence_to_requirements
from langgraph_review_workflow.steps.requirements import extract_requirements_from_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live-model tests unless RUN_LIVE=1 and Ollama is reachable."""

    live_items = [item for item in items if "live" in item.keywords]
    if not live_items:
        return

    if os.environ.get("RUN_LIVE") != "1":
        reason = "live tests need RUN_LIVE=1"
    else:
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        if OllamaClient(model="-", base_url=base_url, timeout_seconds=3).is_available():
            return
        reason = f"Ollama is not reachable at {base_url}"

    for item in live_items:
        item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def candidate_profile() -> dict:
    path = PROJECT_ROOT / "data" / "mock" / "candidate_profile.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def job_description() -> str:
    path = PROJECT_ROOT / "data" / "mock" / "job_description.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def extracted_requirements(job_description: str) -> list[dict]:
    result = extract_requirements_from_text(job_description)
    return [requirement.model_dump(mode="json") for requirement in result.requirements]


@pytest.fixture
def evidence_map(candidate_profile: dict, extracted_requirements: list[dict]) -> list[dict]:
    result = map_evidence_to_requirements(
        {
            "candidate_profile": candidate_profile,
            "requirements": extracted_requirements,
        }
    )
    return [mapping.model_dump(mode="json") for mapping in result.mappings]
