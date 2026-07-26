"""Coverage scoring step."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from langgraph_review_workflow.schemas import (
    CoverageScoringResult,
    CoverageStatus,
    RequirementCoverageScore,
    RequirementEvidenceMap,
    RequirementPriority,
    SupportLevel,
)


PRIORITY_WEIGHTS: dict[RequirementPriority, float] = {
    "must_have": 3.0,
    "should_have": 2.0,
    "nice_to_have": 1.0,
    "policy": 0.0,
}

SUPPORT_MULTIPLIERS: dict[SupportLevel, float] = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "not_supported": 0.0,
    "policy_constraint": 0.0,
}


def _coverage_status(support_level: SupportLevel) -> CoverageStatus:
    if support_level == "supported":
        return "full"
    if support_level == "partially_supported":
        return "partial"
    if support_level == "policy_constraint":
        return "excluded_policy"
    return "gap"


def _gap_summary(mapping: RequirementEvidenceMap, status: CoverageStatus) -> str:
    if status == "excluded_policy":
        return "Policy constraint; excluded from candidate fit score."
    if status == "full":
        return "Requirement is fully covered by explicit evidence."
    if status == "partial":
        missing = ", ".join(mapping.missing_concepts) or "unspecified concepts"
        return f"Partially covered; missing explicit evidence for {missing}."

    missing = ", ".join(mapping.missing_concepts) or "the requirement"
    return f"No explicit supporting evidence for {missing}."


def _fit_summary(
    *,
    total_score: float,
    max_score: float,
    coverage_percent: float,
    unsupported_count: int,
    partial_count: int,
) -> str:
    base = (
        f"Weighted coverage is {total_score:.1f}/{max_score:.1f} "
        f"({coverage_percent:.1f}%)."
    )
    return (
        f"{base} {unsupported_count} requirement(s) have no explicit support; "
        f"{partial_count} requirement(s) are partially supported."
    )


def score_coverage_from_evidence_map(payload: dict[str, Any]) -> CoverageScoringResult:
    """Score requirement coverage from explicit evidence mappings."""

    mappings = [
        RequirementEvidenceMap.model_validate(mapping)
        for mapping in payload.get("evidence_map", [])
    ]
    requirement_scores: list[RequirementCoverageScore] = []

    for mapping in mappings:
        weight = PRIORITY_WEIGHTS[mapping.priority]
        multiplier = SUPPORT_MULTIPLIERS[mapping.support_level]
        max_score = weight
        score = round(weight * multiplier, 2)
        status = _coverage_status(mapping.support_level)
        evidence_ids = [evidence.evidence_id for evidence in mapping.evidence]

        requirement_scores.append(
            RequirementCoverageScore(
                requirement_id=mapping.requirement_id,
                requirement_text=mapping.requirement_text,
                priority=mapping.priority,
                category=mapping.category,
                support_level=mapping.support_level,
                coverage_status=status,
                weight=weight,
                score=score,
                max_score=max_score,
                evidence_ids=evidence_ids,
                gap_summary=_gap_summary(mapping, status),
            )
        )

    total_score = round(sum(score.score for score in requirement_scores), 2)
    max_score = round(sum(score.max_score for score in requirement_scores), 2)
    coverage_percent = round((total_score / max_score) * 100, 1) if max_score else 0.0
    supported_count = sum(score.support_level == "supported" for score in requirement_scores)
    partial_count = sum(
        score.support_level == "partially_supported" for score in requirement_scores
    )
    unsupported_count = sum(
        score.support_level == "not_supported" for score in requirement_scores
    )
    policy_count = sum(
        score.support_level == "policy_constraint" for score in requirement_scores
    )
    gap_requirement_ids = [
        score.requirement_id
        for score in requirement_scores
        if score.coverage_status == "gap"
    ]
    partial_requirement_ids = [
        score.requirement_id
        for score in requirement_scores
        if score.coverage_status == "partial"
    ]

    return CoverageScoringResult(
        requirement_scores=requirement_scores,
        total_score=total_score,
        max_score=max_score,
        coverage_percent=coverage_percent,
        supported_count=supported_count,
        partial_count=partial_count,
        unsupported_count=unsupported_count,
        policy_count=policy_count,
        gap_requirement_ids=gap_requirement_ids,
        partial_requirement_ids=partial_requirement_ids,
        summary=_fit_summary(
            total_score=total_score,
            max_score=max_score,
            coverage_percent=coverage_percent,
            unsupported_count=unsupported_count,
            partial_count=partial_count,
        ),
    )


coverage_scoring_runnable = RunnableLambda(score_coverage_from_evidence_map)
