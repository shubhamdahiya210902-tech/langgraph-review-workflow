"""Live-model groundedness check.

A local model labels each claim as grounded, unsupported, or contradicted using
only the candidate profile. The labels are converted into the same
`GroundednessCheckResult` contract as the deterministic step, so the approval
gate and drafting steps run unchanged downstream.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableLambda

from langgraph_review_workflow.model_boundary import ModelBoundaryConfig
from langgraph_review_workflow.ollama_client import OllamaClient, StructuredChatClient
from langgraph_review_workflow.schemas import (
    ClaimEvidenceSupport,
    ClaimGroundednessCheck,
    GroundednessCheckResult,
    LiveClaimLabel,
    LiveGroundednessLabels,
)
from langgraph_review_workflow.steps.evidence import _snippet_for
from langgraph_review_workflow.steps.groundedness import (
    _claims_to_review,
    _safe_action_for,
    _severity_for,
)


SYSTEM_PROMPT = """You are a strict evidence reviewer for candidate profiles.
Label every claim using ONLY the candidate profile provided. Do not use outside knowledge.

Labels:
- grounded: the profile evidence, skills, or languages directly support the whole claim.
- unsupported: the profile contains no evidence for the claim (or only for part of it).
- contradicted: the profile explicitly states something that conflicts with the claim.

Rules:
- Missing evidence means unsupported, not contradicted. Use contradicted only when the
  profile states a conflicting fact.
- Language levels use the CEFR scale: A1-A2 basic, B1-B2 intermediate, C1-C2 advanced.
  A claim of fluency or expert level in a language listed at A1, A2, or B1 is contradicted.

Return one label per claim, using the claim_id exactly as given. List the supporting
evidence IDs (such as EV-001) only for grounded claims."""


def _profile_context(candidate_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline": candidate_profile.get("headline"),
        "languages": candidate_profile.get("languages", []),
        "skills": candidate_profile.get("skills", []),
        "evidence": candidate_profile.get("evidence", []),
    }


def build_groundedness_prompt(candidate_profile: dict[str, Any]) -> str:
    """Build the user prompt with the profile and numbered claims."""

    claims = [
        {"claim_id": f"CLAIM-{index:03d}", "claim": claim}
        for index, claim in enumerate(_claims_to_review(candidate_profile), start=1)
    ]
    return (
        "Candidate profile:\n"
        f"{json.dumps(_profile_context(candidate_profile), indent=2)}\n\n"
        "Claims to label:\n"
        f"{json.dumps(claims, indent=2)}"
    )


def _evidence_supports(
    evidence_ids: list[str],
    candidate_profile: dict[str, Any],
) -> list[ClaimEvidenceSupport]:
    evidence_by_id = {
        str(evidence["id"]): evidence for evidence in candidate_profile.get("evidence", [])
    }
    supports: list[ClaimEvidenceSupport] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None or any(s.evidence_id == evidence_id for s in supports):
            continue
        supports.append(
            ClaimEvidenceSupport(
                evidence_id=evidence_id,
                title=str(evidence.get("title", evidence_id)),
                evidence_type=str(evidence.get("type", "evidence")),
                support_snippet=_snippet_for(evidence),
            )
        )
    return supports


def labels_to_result(
    labels: LiveGroundednessLabels,
    candidate_profile: dict[str, Any],
) -> GroundednessCheckResult:
    """Convert live labels into the shared groundedness contract.

    A claim the model skipped is treated as unsupported, so a missing label can
    never let a claim through the approval gate.
    """

    labels_by_id: dict[str, LiveClaimLabel] = {}
    for label in labels.labels:
        labels_by_id.setdefault(label.claim_id.strip().upper(), label)

    checks: list[ClaimGroundednessCheck] = []
    for index, claim in enumerate(_claims_to_review(candidate_profile), start=1):
        claim_id = f"CLAIM-{index:03d}"
        label = labels_by_id.get(claim_id)
        status = label.status if label else "unsupported"
        explanation = label.rationale if label else "The live model returned no label."
        evidence = (
            _evidence_supports(label.evidence_ids, candidate_profile)
            if label and status == "grounded"
            else []
        )
        checks.append(
            ClaimGroundednessCheck(
                claim_id=claim_id,
                claim=claim,
                status=status,
                severity=_severity_for(status),
                evidence=evidence,
                explanation=explanation,
                safe_action=_safe_action_for(status),
            )
        )

    unsupported_claims = [check for check in checks if check.status != "grounded"]
    grounded_count = sum(check.status == "grounded" for check in checks)
    unsupported_count = sum(check.status == "unsupported" for check in checks)
    contradicted_count = sum(check.status == "contradicted" for check in checks)

    return GroundednessCheckResult(
        checks=checks,
        unsupported_claims=unsupported_claims,
        grounded_count=grounded_count,
        partial_count=0,
        unsupported_count=unsupported_count,
        contradicted_count=contradicted_count,
        summary=(
            f"Live model checked {len(checks)} claim(s): {grounded_count} grounded, "
            f"{unsupported_count} unsupported, {contradicted_count} contradicted."
        ),
    )


def check_claim_groundedness_live(
    payload: dict[str, Any],
    client: StructuredChatClient | None = None,
) -> GroundednessCheckResult:
    """Label claims with a live model and return the shared groundedness contract."""

    boundary = ModelBoundaryConfig.model_validate(payload["model_boundary"])
    candidate_profile = payload["candidate_profile"]
    client = client or OllamaClient.from_boundary(boundary)

    labels = client.structured_chat(
        system=SYSTEM_PROMPT,
        user=build_groundedness_prompt(candidate_profile),
        schema=LiveGroundednessLabels,
        temperature=boundary.temperature,
        seed=boundary.seed,
    )
    return labels_to_result(labels, candidate_profile)


live_groundedness_check_runnable = RunnableLambda(check_claim_groundedness_live)
