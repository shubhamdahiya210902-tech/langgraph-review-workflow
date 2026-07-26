"""Shared state types for the review workflow."""

from typing import Any, Literal, TypedDict


ApprovalStatus = Literal["pending", "approved", "rejected"]
StepStatus = Literal["complete", "skipped"]


class AuditEvent(TypedDict):
    """Human-readable trace event for demo and debugging output."""

    node: str
    status: StepStatus
    message: str


class ReviewState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    Intermediate outputs are stored as dictionaries so the CLI can display
    structured Pydantic results without custom serialization.
    """

    candidate_profile_path: str
    candidate_profile: dict[str, Any]
    job_description_path: str
    job_description: str
    role_title: str
    company: str | None
    model_boundary: dict[str, Any]

    requirements: list[dict[str, Any]]
    evidence_map: list[dict[str, Any]]
    coverage_scores: list[dict[str, Any]]
    coverage_summary: dict[str, Any]
    groundedness_checks: list[dict[str, Any]]
    groundedness_summary: dict[str, Any]
    unsupported_claims: list[dict[str, Any]]

    human_approved: bool
    approval_status: ApprovalStatus
    approval_decision: dict[str, Any]
    final_draft: str
    final_draft_metadata: dict[str, Any]

    audit_trail: list[AuditEvent]
