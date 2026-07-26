"""Evidence mapping step.

The mapper is deterministic and evidence-bound: it only uses the supplied mock
candidate profile, extracted requirements, and explicit concept aliases.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.runnables import RunnableLambda

from langgraph_review_workflow.schemas import (
    EvidenceMappingResult,
    EvidenceMatch,
    ExtractedRequirement,
    RequirementEvidenceMap,
    SupportLevel,
)


CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "workflow_components": ("workflow", "components"),
    "text_review": ("responses", "answers", "text inputs"),
    "structured_schemas": ("structured", "schema", "schemas", "pydantic", "models"),
    "candidate_evidence": ("candidate evidence", "source ids", "citation", "snippets"),
    "coverage_scoring": ("scoring", "rubric", "rubrics", "criteria"),
    "unsupported_claim_checks": (
        "unsupported",
        "speculative",
        "flagged",
        "supporting snippet",
        "no supporting",
    ),
    "pytest": ("pytest", "tests", "test coverage", "unit tests"),
    "documentation": ("document", "documented", "documentation", "technical writing", "setup"),
    "api": ("api", "apis", "fastapi"),
    "data_validation": ("validation", "validate", "validated"),
    "llm_prompting": ("llm", "prompt", "prompting"),
    "retrieval_augmented_generation": ("retrieval", "rag", "augmented generation"),
    "git": ("git",),
    "langgraph": ("langgraph",),
    "langchain": ("langchain",),
    "workflow_orchestration": ("workflow orchestration", "orchestration"),
    "fastapi": ("fastapi",),
    "backend_services": ("backend", "service", "services"),
    "human_in_the_loop": ("human-in-the-loop", "human in the loop", "human approval"),
    "english": ("english",),
    "german": ("german",),
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _contains_alias(text: str, alias: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_alias = _normalize_text(alias)
    if " " in normalized_alias or "-" in normalized_alias:
        return normalized_alias in normalized_text
    return bool(re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_text))


def _concepts_in_text(text: str) -> list[str]:
    concepts: list[str] = []

    for concept, aliases in CONCEPT_ALIASES.items():
        if any(_contains_alias(text, alias) for alias in aliases):
            concepts.append(concept)

    return concepts


def _evidence_text(evidence: dict[str, Any]) -> str:
    parts = [
        str(evidence.get("title", "")),
        str(evidence.get("summary", "")),
        " ".join(str(skill) for skill in evidence.get("skills", [])),
        " ".join(str(outcome) for outcome in evidence.get("outcomes", [])),
    ]
    return " ".join(part for part in parts if part)


def _snippet_for(evidence: dict[str, Any]) -> str:
    outcomes = evidence.get("outcomes", [])
    if outcomes:
        return str(outcomes[0])
    return str(evidence.get("summary", ""))


def _profile_language_matches(
    requirement: ExtractedRequirement,
    candidate_profile: dict[str, Any],
) -> list[EvidenceMatch]:
    requirement_concepts = set(_concepts_in_text(requirement.text))
    matches: list[EvidenceMatch] = []

    for language in candidate_profile.get("languages", []):
        language_name = str(language.get("language", ""))
        concept = language_name.lower()
        if concept not in requirement_concepts:
            continue

        matches.append(
            EvidenceMatch(
                evidence_id=f"PROFILE-LANG-{language_name}",
                title=f"{language_name} language profile",
                evidence_type="profile_language",
                matched_concepts=[concept],
                support_snippet=f"{language_name}: {language.get('level', 'not specified')}",
            )
        )

    return matches


def _support_level(
    *,
    matched_concepts: list[str],
    missing_concepts: list[str],
) -> SupportLevel:
    if not matched_concepts:
        return "not_supported"
    if not missing_concepts:
        return "supported"
    return "partially_supported"


def _rationale_for(
    *,
    support_level: SupportLevel,
    matched_concepts: list[str],
    missing_concepts: list[str],
) -> str:
    if support_level == "policy_constraint":
        return "This requirement is a claim-support policy for later groundedness checks."
    if support_level == "supported":
        return f"Explicit candidate evidence covers: {', '.join(matched_concepts)}."
    if support_level == "partially_supported":
        return (
            f"Candidate evidence covers {', '.join(matched_concepts)}; "
            f"still missing {', '.join(missing_concepts)}."
        )
    return f"No explicit candidate evidence found for: {', '.join(missing_concepts)}."


def map_evidence_to_requirements(payload: dict[str, Any]) -> EvidenceMappingResult:
    """Map extracted requirements to explicit candidate evidence."""

    candidate_profile = payload["candidate_profile"]
    requirements = [
        ExtractedRequirement.model_validate(requirement)
        for requirement in payload.get("requirements", [])
    ]
    profile_evidence = candidate_profile.get("evidence", [])
    mappings: list[RequirementEvidenceMap] = []

    for requirement in requirements:
        if requirement.category == "claim_policy":
            mappings.append(
                RequirementEvidenceMap(
                    requirement_id=requirement.requirement_id,
                    requirement_text=requirement.text,
                    priority=requirement.priority,
                    category=requirement.category,
                    support_level="policy_constraint",
                    rationale=_rationale_for(
                        support_level="policy_constraint",
                        matched_concepts=[],
                        missing_concepts=[],
                    ),
                )
            )
            continue

        requirement_concepts = _concepts_in_text(requirement.text)
        evidence_matches: list[EvidenceMatch] = []
        matched_concepts: list[str] = []

        for evidence in profile_evidence:
            evidence_concepts = _concepts_in_text(_evidence_text(evidence))
            shared_concepts = [
                concept for concept in requirement_concepts if concept in evidence_concepts
            ]
            if not shared_concepts:
                continue

            for concept in shared_concepts:
                if concept not in matched_concepts:
                    matched_concepts.append(concept)

            evidence_matches.append(
                EvidenceMatch(
                    evidence_id=str(evidence["id"]),
                    title=str(evidence.get("title", evidence["id"])),
                    evidence_type=str(evidence.get("type", "evidence")),
                    matched_concepts=shared_concepts,
                    support_snippet=_snippet_for(evidence),
                )
            )

        for language_match in _profile_language_matches(requirement, candidate_profile):
            for concept in language_match.matched_concepts:
                if concept not in matched_concepts:
                    matched_concepts.append(concept)
            evidence_matches.append(language_match)

        missing_concepts = [
            concept for concept in requirement_concepts if concept not in matched_concepts
        ]
        support_level = _support_level(
            matched_concepts=matched_concepts,
            missing_concepts=missing_concepts,
        )

        mappings.append(
            RequirementEvidenceMap(
                requirement_id=requirement.requirement_id,
                requirement_text=requirement.text,
                priority=requirement.priority,
                category=requirement.category,
                support_level=support_level,
                matched_concepts=matched_concepts,
                missing_concepts=missing_concepts,
                evidence=evidence_matches,
                rationale=_rationale_for(
                    support_level=support_level,
                    matched_concepts=matched_concepts,
                    missing_concepts=missing_concepts,
                ),
            )
        )

    return EvidenceMappingResult(mappings=mappings)


evidence_mapping_runnable = RunnableLambda(map_evidence_to_requirements)
