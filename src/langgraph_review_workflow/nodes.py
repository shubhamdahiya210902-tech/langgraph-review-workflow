"""LangGraph node functions for the review workflow."""

from langgraph_review_workflow.state import AuditEvent, ReviewState, StepStatus
from langgraph_review_workflow.steps.approval import approval_gate_runnable
from langgraph_review_workflow.steps.drafting import drafting_runnable
from langgraph_review_workflow.steps.evidence import evidence_mapping_runnable
from langgraph_review_workflow.steps.groundedness import groundedness_check_runnable
from langgraph_review_workflow.steps.requirements import requirement_extraction_runnable
from langgraph_review_workflow.steps.scoring import coverage_scoring_runnable


def _with_audit(
    state: ReviewState,
    *,
    node: str,
    message: str,
    status: StepStatus = "complete",
    **updates: object,
) -> ReviewState:
    audit_event: AuditEvent = {
        "node": node,
        "status": status,
        "message": message,
    }
    return {
        **updates,
        "audit_trail": [*state.get("audit_trail", []), audit_event],
    }


def load_inputs(state: ReviewState) -> ReviewState:
    """Confirm that the CLI supplied the required mock inputs."""

    has_candidate = bool(state.get("candidate_profile"))
    has_job_description = bool(state.get("job_description"))
    status: StepStatus = "complete" if has_candidate and has_job_description else "skipped"
    message = (
        "Mock candidate profile and job description loaded."
        if status == "complete"
        else "Missing candidate profile or job description."
    )
    return _with_audit(state, node="load_inputs", status=status, message=message)


def extract_requirements(state: ReviewState) -> ReviewState:
    """Extract structured requirements from the job description."""

    extraction = requirement_extraction_runnable.invoke(state["job_description"])
    requirements = [
        requirement.model_dump(mode="json") for requirement in extraction.requirements
    ]

    return _with_audit(
        state,
        node="extract_requirements",
        status="complete",
        message=(
            f"Extracted {len(requirements)} structured requirements "
            f"for {extraction.role_title}."
        ),
        role_title=extraction.role_title,
        company=extraction.company,
        requirements=requirements,
    )


def map_evidence(state: ReviewState) -> ReviewState:
    """Map candidate evidence to extracted requirements."""

    mapping_result = evidence_mapping_runnable.invoke(
        {
            "candidate_profile": state["candidate_profile"],
            "requirements": state.get("requirements", []),
        }
    )
    evidence_map = [mapping.model_dump(mode="json") for mapping in mapping_result.mappings]
    supported_count = sum(
        1 for mapping in evidence_map if mapping["support_level"] == "supported"
    )
    partial_count = sum(
        1 for mapping in evidence_map if mapping["support_level"] == "partially_supported"
    )
    unsupported_count = sum(
        1 for mapping in evidence_map if mapping["support_level"] == "not_supported"
    )

    return _with_audit(
        state,
        node="map_evidence",
        status="complete",
        message=(
            "Mapped evidence for "
            f"{len(evidence_map)} requirements: {supported_count} supported, "
            f"{partial_count} partial, {unsupported_count} unsupported."
        ),
        evidence_map=evidence_map,
    )


def score_coverage(state: ReviewState) -> ReviewState:
    """Score requirement coverage with an explainable rubric."""

    scoring_result = coverage_scoring_runnable.invoke(
        {"evidence_map": state.get("evidence_map", [])}
    )
    coverage_scores = [
        score.model_dump(mode="json") for score in scoring_result.requirement_scores
    ]
    coverage_summary = scoring_result.model_dump(
        mode="json",
        exclude={"requirement_scores"},
    )

    return _with_audit(
        state,
        node="score_coverage",
        status="complete",
        message=(
            "Scored coverage at "
            f"{scoring_result.coverage_percent:.1f}% weighted fit "
            f"({scoring_result.total_score:.1f}/{scoring_result.max_score:.1f})."
        ),
        coverage_scores=coverage_scores,
        coverage_summary=coverage_summary,
    )


def check_unsupported_claims(state: ReviewState) -> ReviewState:
    """Check proposed claims for evidence-grounded support."""

    groundedness_result = groundedness_check_runnable.invoke(
        {"candidate_profile": state["candidate_profile"]}
    )
    groundedness_checks = [
        check.model_dump(mode="json") for check in groundedness_result.checks
    ]
    unsupported_claims = [
        claim.model_dump(mode="json") for claim in groundedness_result.unsupported_claims
    ]
    groundedness_summary = groundedness_result.model_dump(
        mode="json",
        exclude={"checks", "unsupported_claims"},
    )

    return _with_audit(
        state,
        node="check_unsupported_claims",
        status="complete",
        message=groundedness_result.summary,
        groundedness_checks=groundedness_checks,
        unsupported_claims=unsupported_claims,
        groundedness_summary=groundedness_summary,
    )


def human_approval_gate(state: ReviewState) -> ReviewState:
    """Gate final drafting on explicit human approval."""

    approval_result = approval_gate_runnable.invoke(
        {
            "human_approved": state.get("human_approved", False),
            "groundedness_checks": state.get("groundedness_checks", []),
            "unsupported_claims": state.get("unsupported_claims", []),
        }
    )
    return _with_audit(
        state,
        node="human_approval_gate",
        status="complete" if approval_result.can_draft else "skipped",
        message=approval_result.summary,
        approval_status=approval_result.approval_status,
        approval_decision=approval_result.model_dump(mode="json"),
    )


def draft_final_response(state: ReviewState) -> ReviewState:
    """Generate the safe final draft after human approval."""

    draft_result = drafting_runnable.invoke(
        {
            "approval_decision": state["approval_decision"],
            "groundedness_checks": state.get("groundedness_checks", []),
            "coverage_summary": state.get("coverage_summary", {}),
            "candidate_profile": state.get("candidate_profile", {}),
            "role_title": state.get("role_title", "Mock Role"),
        }
    )

    return _with_audit(
        state,
        node="draft_final_response",
        status="complete",
        message=(
            "Generated a safe final draft with "
            f"{len(draft_result.included_claim_ids)} grounded claim(s) and "
            f"{len(draft_result.excluded_claim_ids)} excluded claim(s)."
        ),
        final_draft=draft_result.draft_text,
        final_draft_metadata=draft_result.model_dump(
            mode="json",
            exclude={"draft_text"},
        ),
    )
