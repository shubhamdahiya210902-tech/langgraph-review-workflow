from langgraph_review_workflow.graph import build_graph
from langgraph_review_workflow.steps.approval import run_human_approval_gate
from langgraph_review_workflow.steps.drafting import generate_safe_final_draft
from langgraph_review_workflow.steps.groundedness import check_claim_groundedness
from langgraph_review_workflow.steps.scoring import score_coverage_from_evidence_map


def test_approval_gate_requires_human_confirmation(candidate_profile: dict) -> None:
    groundedness = check_claim_groundedness({"candidate_profile": candidate_profile})
    checks = [check.model_dump(mode="json") for check in groundedness.checks]

    pending = run_human_approval_gate(
        {"human_approved": False, "groundedness_checks": checks}
    )
    approved = run_human_approval_gate(
        {"human_approved": True, "groundedness_checks": checks}
    )

    assert pending.approval_status == "pending"
    assert pending.can_draft is False
    assert pending.approved_claim_ids == []
    assert approved.approval_status == "approved"
    assert approved.can_draft is True
    assert approved.approved_claim_ids == ["CLAIM-001", "CLAIM-002", "CLAIM-003"]
    assert approved.blocked_claim_ids == [
        "CLAIM-004",
        "CLAIM-005",
        "CLAIM-006",
        "CLAIM-007",
    ]


def test_final_draft_uses_only_approved_grounded_claims(
    candidate_profile: dict,
    evidence_map: list[dict],
) -> None:
    groundedness = check_claim_groundedness({"candidate_profile": candidate_profile})
    checks = [check.model_dump(mode="json") for check in groundedness.checks]
    approval = run_human_approval_gate(
        {"human_approved": True, "groundedness_checks": checks}
    )
    scoring = score_coverage_from_evidence_map({"evidence_map": evidence_map})
    draft = generate_safe_final_draft(
        {
            "approval_decision": approval.model_dump(mode="json"),
            "groundedness_checks": checks,
            "coverage_summary": scoring.model_dump(
                mode="json",
                exclude={"requirement_scores"},
            ),
            "candidate_profile": candidate_profile,
            "role_title": "Junior AI Workflow Engineer - Mock Role",
        }
    )

    assert draft.included_claim_ids == ["CLAIM-001", "CLAIM-002", "CLAIM-003"]
    assert draft.excluded_claim_ids == [
        "CLAIM-004",
        "CLAIM-005",
        "CLAIM-006",
        "CLAIM-007",
    ]
    assert "Built a mock RAG support assistant" in draft.draft_text
    assert "Production LangGraph deployment experience" not in draft.draft_text
    assert "Led a team of five engineers" not in draft.draft_text
    assert "Expert-level German fluency" not in draft.draft_text
    assert "Certified cloud architect" not in draft.draft_text


def test_graph_routes_pending_and_approved_final_draft(
    candidate_profile: dict,
    job_description: str,
) -> None:
    base_state = {
        "candidate_profile": candidate_profile,
        "job_description": job_description,
        "audit_trail": [],
    }

    pending = build_graph().invoke({**base_state, "human_approved": False})
    approved = build_graph().invoke({**base_state, "human_approved": True})

    assert pending["approval_status"] == "pending"
    assert "final_draft" not in pending
    assert approved["approval_status"] == "approved"
    assert approved["approval_decision"]["approved_claim_ids"] == [
        "CLAIM-001",
        "CLAIM-002",
        "CLAIM-003",
    ]
    assert "# Evidence-Grounded Candidate Review Draft" in approved["final_draft"]
    assert "Production LangGraph deployment experience" not in approved["final_draft"]
