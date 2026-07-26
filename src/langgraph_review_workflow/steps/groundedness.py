"""Unsupported-claim and groundedness checks."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from langgraph_review_workflow.schemas import (
    ClaimEvidenceSupport,
    ClaimGroundednessCheck,
    ClaimSeverity,
    ClaimStatus,
    GroundednessCheckResult,
)
from langgraph_review_workflow.steps.evidence import (
    _concepts_in_text,
    _contains_alias,
    _evidence_text,
    _snippet_for,
)


LOW_LANGUAGE_LEVELS = {"a1", "a2", "b1"}


def _append_unique(items: list[str], new_items: list[str]) -> None:
    for item in new_items:
        if item not in items:
            items.append(item)


def _extra_claim_concepts(text: str) -> list[str]:
    concepts: list[str] = []

    if _contains_alias(text, "production") and any(
        _contains_alias(text, alias) for alias in ("deployment", "deployed")
    ):
        concepts.append("production_deployment")

    if any(_contains_alias(text, alias) for alias in ("led", "lead", "managed")) and any(
        _contains_alias(text, alias) for alias in ("team", "engineers")
    ):
        concepts.append("team_leadership")

    if any(
        _contains_alias(text, alias)
        for alias in ("expert-level", "expert level", "fluent", "fluency", "native")
    ):
        concepts.append("expert_language")

    if any(
        _contains_alias(text, alias)
        for alias in ("certified", "certification", "cloud architect")
    ):
        concepts.append("cloud_certification")

    return concepts


def _claim_concepts(text: str) -> list[str]:
    concepts = _concepts_in_text(text)
    _append_unique(concepts, _extra_claim_concepts(text))
    return concepts


def _profile_language_support(
    *,
    claim: str,
    claim_concepts: list[str],
    candidate_profile: dict[str, Any],
) -> list[ClaimEvidenceSupport]:
    supports: list[ClaimEvidenceSupport] = []

    for language in candidate_profile.get("languages", []):
        language_name = str(language.get("language", ""))
        concept = language_name.lower()
        if concept not in claim_concepts or not _contains_alias(claim, language_name):
            continue

        supports.append(
            ClaimEvidenceSupport(
                evidence_id=f"PROFILE-LANG-{language_name}",
                title=f"{language_name} language profile",
                evidence_type="profile_language",
                matched_concepts=[concept],
                support_snippet=f"{language_name}: {language.get('level', 'not specified')}",
            )
        )

    return supports


def _profile_skill_support(
    *,
    claim_concepts: list[str],
    candidate_profile: dict[str, Any],
) -> list[ClaimEvidenceSupport]:
    skill_text = " ".join(str(skill) for skill in candidate_profile.get("skills", []))
    skill_concepts = _concepts_in_text(skill_text)
    matched_concepts = [
        concept for concept in claim_concepts if concept in skill_concepts
    ]

    if not matched_concepts:
        return []

    return [
        ClaimEvidenceSupport(
            evidence_id="PROFILE-SKILLS",
            title="Candidate skills list",
            evidence_type="profile_skills",
            matched_concepts=matched_concepts,
            support_snippet=", ".join(str(skill) for skill in candidate_profile.get("skills", [])),
        )
    ]


def _evidence_support(
    *,
    claim_concepts: list[str],
    candidate_profile: dict[str, Any],
) -> list[ClaimEvidenceSupport]:
    supports: list[ClaimEvidenceSupport] = []

    for evidence in candidate_profile.get("evidence", []):
        evidence_concepts = _concepts_in_text(_evidence_text(evidence))
        matched_concepts = [
            concept for concept in claim_concepts if concept in evidence_concepts
        ]
        if not matched_concepts:
            continue

        supports.append(
            ClaimEvidenceSupport(
                evidence_id=str(evidence["id"]),
                title=str(evidence.get("title", evidence["id"])),
                evidence_type=str(evidence.get("type", "evidence")),
                matched_concepts=matched_concepts,
                support_snippet=_snippet_for(evidence),
            )
        )

    return supports


def _has_language_contradiction(
    *,
    claim_concepts: list[str],
    candidate_profile: dict[str, Any],
) -> bool:
    if "expert_language" not in claim_concepts:
        return False

    for language in candidate_profile.get("languages", []):
        language_name = str(language.get("language", "")).lower()
        language_level = str(language.get("level", "")).lower()
        if language_name in claim_concepts and language_level in LOW_LANGUAGE_LEVELS:
            return True

    return False


def _status_for(
    *,
    claim_concepts: list[str],
    matched_concepts: list[str],
    contradicted: bool,
) -> ClaimStatus:
    if contradicted:
        return "contradicted"
    if not claim_concepts or not matched_concepts:
        return "unsupported"
    if set(claim_concepts).issubset(set(matched_concepts)):
        return "grounded"
    return "partially_grounded"


def _severity_for(status: ClaimStatus) -> ClaimSeverity:
    if status == "grounded":
        return "none"
    if status == "partially_grounded":
        return "low"
    if status == "contradicted":
        return "high"
    return "medium"


def _safe_action_for(status: ClaimStatus) -> str:
    if status == "grounded":
        return "Allow claim with cited evidence."
    if status == "partially_grounded":
        return "Revise to include only supported concepts before drafting."
    if status == "contradicted":
        return "Block claim; it conflicts with explicit profile evidence."
    return "Block claim until explicit supporting evidence is added."


def _explanation_for(
    *,
    status: ClaimStatus,
    matched_concepts: list[str],
    missing_concepts: list[str],
) -> str:
    if status == "grounded":
        return f"All detected claim concepts are supported: {', '.join(matched_concepts)}."
    if status == "partially_grounded":
        return (
            f"Supported concepts: {', '.join(matched_concepts)}. "
            f"Missing support for: {', '.join(missing_concepts)}."
        )
    if status == "contradicted":
        return (
            "The claim includes an expert-level language assertion that conflicts "
            "with the explicit profile language level."
        )
    if missing_concepts:
        return f"No explicit evidence supports: {', '.join(missing_concepts)}."
    return "No recognizable supported evidence concepts were found in this claim."


def _claims_to_review(candidate_profile: dict[str, Any]) -> list[str]:
    claims = candidate_profile.get("claims_to_review")
    if claims:
        return [str(claim) for claim in claims]

    return [str(claim) for claim in candidate_profile.get("unsupported_claim_examples", [])]


def check_claim_groundedness(payload: dict[str, Any]) -> GroundednessCheckResult:
    """Check whether proposed candidate-facing claims are evidence-grounded."""

    candidate_profile = payload["candidate_profile"]
    checks: list[ClaimGroundednessCheck] = []

    for index, claim in enumerate(_claims_to_review(candidate_profile), start=1):
        claim_concepts = _claim_concepts(claim)
        evidence = [
            *_evidence_support(
                claim_concepts=claim_concepts,
                candidate_profile=candidate_profile,
            ),
            *_profile_language_support(
                claim=claim,
                claim_concepts=claim_concepts,
                candidate_profile=candidate_profile,
            ),
            *_profile_skill_support(
                claim_concepts=claim_concepts,
                candidate_profile=candidate_profile,
            ),
        ]
        matched_concepts: list[str] = []
        for support in evidence:
            _append_unique(matched_concepts, support.matched_concepts)

        missing_concepts = [
            concept for concept in claim_concepts if concept not in matched_concepts
        ]
        contradicted = _has_language_contradiction(
            claim_concepts=claim_concepts,
            candidate_profile=candidate_profile,
        )
        status = _status_for(
            claim_concepts=claim_concepts,
            matched_concepts=matched_concepts,
            contradicted=contradicted,
        )

        checks.append(
            ClaimGroundednessCheck(
                claim_id=f"CLAIM-{index:03d}",
                claim=claim,
                status=status,
                severity=_severity_for(status),
                matched_concepts=matched_concepts,
                missing_concepts=missing_concepts,
                evidence=evidence,
                explanation=_explanation_for(
                    status=status,
                    matched_concepts=matched_concepts,
                    missing_concepts=missing_concepts,
                ),
                safe_action=_safe_action_for(status),
            )
        )

    unsupported_claims = [check for check in checks if check.status != "grounded"]
    grounded_count = sum(check.status == "grounded" for check in checks)
    partial_count = sum(check.status == "partially_grounded" for check in checks)
    unsupported_count = sum(check.status == "unsupported" for check in checks)
    contradicted_count = sum(check.status == "contradicted" for check in checks)
    summary = (
        f"Checked {len(checks)} claim(s): {grounded_count} grounded, "
        f"{partial_count} partial, {unsupported_count} unsupported, "
        f"{contradicted_count} contradicted."
    )

    return GroundednessCheckResult(
        checks=checks,
        unsupported_claims=unsupported_claims,
        grounded_count=grounded_count,
        partial_count=partial_count,
        unsupported_count=unsupported_count,
        contradicted_count=contradicted_count,
        summary=summary,
    )


groundedness_check_runnable = RunnableLambda(check_claim_groundedness)
