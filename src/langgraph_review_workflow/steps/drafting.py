"""Safe final draft generation."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from langgraph_review_workflow.schemas import ClaimGroundednessCheck, FinalDraftResult


def _evidence_ids_for(check: ClaimGroundednessCheck) -> list[str]:
    ids: list[str] = []
    for evidence in check.evidence:
        if evidence.evidence_id not in ids:
            ids.append(evidence.evidence_id)
    return ids


def _checks_by_id(checks: list[ClaimGroundednessCheck]) -> dict[str, ClaimGroundednessCheck]:
    return {check.claim_id: check for check in checks}


def _format_claim_bullet(check: ClaimGroundednessCheck) -> str:
    evidence_ids = ", ".join(_evidence_ids_for(check)) or "evidence unavailable"
    return f"- {check.claim} Evidence: {evidence_ids}."


def _format_gap_bullets(coverage_summary: dict[str, Any]) -> list[str]:
    gaps = coverage_summary.get("gap_requirement_ids", [])
    partials = coverage_summary.get("partial_requirement_ids", [])
    bullets: list[str] = []

    if gaps:
        bullets.append(f"- Requirements with no explicit support: {', '.join(gaps)}.")
    if partials:
        bullets.append(f"- Partially supported requirements: {', '.join(partials)}.")
    if not bullets:
        bullets.append("- No uncovered requirements were detected by the scoring rubric.")

    return bullets


def generate_safe_final_draft(payload: dict[str, Any]) -> FinalDraftResult:
    """Generate a safe, evidence-cited draft from approved grounded claims only."""

    approval_decision = payload["approval_decision"]
    checks = [
        ClaimGroundednessCheck.model_validate(check)
        for check in payload.get("groundedness_checks", [])
    ]
    checks_by_id = _checks_by_id(checks)
    included_claim_ids = list(approval_decision.get("approved_claim_ids", []))
    excluded_claim_ids = list(approval_decision.get("blocked_claim_ids", []))
    included_checks = [
        checks_by_id[claim_id]
        for claim_id in included_claim_ids
        if claim_id in checks_by_id and checks_by_id[claim_id].status == "grounded"
    ]
    candidate_profile = payload.get("candidate_profile", {})
    candidate_name = candidate_profile.get("name", "Synthetic Candidate")
    role_title = payload.get("role_title", "Mock Role")
    coverage_summary = payload.get("coverage_summary", {})
    coverage_text = coverage_summary.get("summary", "Coverage summary unavailable.")

    cited_evidence_ids: list[str] = []
    for check in included_checks:
        for evidence_id in _evidence_ids_for(check):
            if evidence_id not in cited_evidence_ids:
                cited_evidence_ids.append(evidence_id)

    warnings = [
        "Synthetic demo output only; not a real job application.",
        "Unsupported and contradicted claims were excluded from the draft.",
        "Final human review is still required before reuse in any real context.",
    ]

    claim_lines = [_format_claim_bullet(check) for check in included_checks]
    if not claim_lines:
        claim_lines = ["- No grounded claims were approved for drafting."]

    blocked_line = (
        ", ".join(excluded_claim_ids)
        if excluded_claim_ids
        else "No blocked claims in this run."
    )
    draft_lines = [
        "# Evidence-Grounded Candidate Review Draft",
        "",
        f"- Candidate: {candidate_name}",
        f"- Role: {role_title}",
        "",
        "## Fit Summary",
        coverage_text,
        "",
        "## Evidence-Grounded Highlights",
        *claim_lines,
        "",
        "## Known Gaps",
        *_format_gap_bullets(coverage_summary),
        "",
        "## Excluded Claims",
        f"- Excluded from drafting: {blocked_line}.",
        "",
        "## Drafting Constraints",
        "- Use only the cited evidence above.",
        "- Do not add claims about production LangGraph deployments, leadership,",
        "  expert German fluency, certifications, or other unsupported experience.",
    ]

    return FinalDraftResult(
        draft_text="\n".join(draft_lines),
        included_claim_ids=included_claim_ids,
        excluded_claim_ids=excluded_claim_ids,
        cited_evidence_ids=cited_evidence_ids,
        warnings=warnings,
    )


drafting_runnable = RunnableLambda(generate_safe_final_draft)
