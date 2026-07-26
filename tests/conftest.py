import json
from pathlib import Path

import pytest

from langgraph_review_workflow.steps.evidence import map_evidence_to_requirements
from langgraph_review_workflow.steps.requirements import extract_requirements_from_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
