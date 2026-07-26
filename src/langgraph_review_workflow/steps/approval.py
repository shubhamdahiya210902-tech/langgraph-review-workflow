"""Human approval gate for final drafting."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from langgraph_review_workflow.schemas import ApprovalGateResult, ClaimGroundednessCheck


SAFE_DRAFT_CONSTRAINTS = [
    "Use only claims with grounded status.",
    "Exclude unsupported, partially grounded, and contradicted claims.",
    "Preserve evidence IDs so every candidate-facing claim remains traceable.",
    "Do not strengthen language beyond the supplied evidence.",
]


def _claim_ids_by_status(
    checks: list[ClaimGroundednessCheck],
    status: str,
) -> list[str]:
    return [check.claim_id for check in checks if check.status == status]


def _blocked_claim_ids(checks: list[ClaimGroundednessCheck]) -> list[str]:
    return [check.claim_id for check in checks if check.status != "grounded"]


def run_human_approval_gate(payload: dict[str, Any]) -> ApprovalGateResult:
    """Require explicit human approval before safe-only final drafting."""

    human_approved = bool(payload.get("human_approved", False))
    checks = [
        ClaimGroundednessCheck.model_validate(check)
        for check in payload.get("groundedness_checks", [])
    ]
    grounded_claim_ids = _claim_ids_by_status(checks, "grounded")
    blocked_claim_ids = _blocked_claim_ids(checks)

    if not human_approved:
        return ApprovalGateResult(
            approval_status="pending",
            can_draft=False,
            human_approved=False,
            grounded_claim_ids=grounded_claim_ids,
            blocked_claim_ids=blocked_claim_ids,
            draft_constraints=SAFE_DRAFT_CONSTRAINTS,
            required_human_action=(
                "Review coverage, blocked claims, and grounded claim evidence. "
                "Re-run with --approve-final to approve safe-only drafting."
            ),
            summary=(
                "Final drafting is pending human approval. "
                f"{len(blocked_claim_ids)} blocked claim(s) must remain excluded."
            ),
        )

    if not grounded_claim_ids:
        return ApprovalGateResult(
            approval_status="rejected",
            can_draft=False,
            human_approved=True,
            grounded_claim_ids=[],
            blocked_claim_ids=blocked_claim_ids,
            draft_constraints=SAFE_DRAFT_CONSTRAINTS,
            required_human_action=(
                "Add explicit evidence for at least one safe claim before drafting."
            ),
            summary="Human approval was provided, but no grounded claims are available.",
        )

    return ApprovalGateResult(
        approval_status="approved",
        can_draft=True,
        human_approved=True,
        grounded_claim_ids=grounded_claim_ids,
        approved_claim_ids=grounded_claim_ids,
        blocked_claim_ids=blocked_claim_ids,
        draft_constraints=SAFE_DRAFT_CONSTRAINTS,
        required_human_action="No further action required before safe-only drafting.",
        summary=(
            "Human approval received for safe-only drafting with "
            f"{len(grounded_claim_ids)} grounded claim(s). "
            f"{len(blocked_claim_ids)} blocked claim(s) remain excluded."
        ),
    )


approval_gate_runnable = RunnableLambda(run_human_approval_gate)
