"""LangGraph workflow assembly."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from langgraph_review_workflow.nodes import (
    check_unsupported_claims,
    draft_final_response,
    extract_requirements,
    human_approval_gate,
    load_inputs,
    map_evidence,
    score_coverage,
)
from langgraph_review_workflow.state import ReviewState


def route_after_human_gate(state: ReviewState) -> Literal["draft_final_response", "end"]:
    """Continue only when a human has approved final drafting."""

    return "draft_final_response" if state.get("approval_status") == "approved" else "end"


def build_graph():
    """Build and compile the review workflow graph."""

    workflow = StateGraph(ReviewState)

    workflow.add_node("load_inputs", load_inputs)
    workflow.add_node("extract_requirements", extract_requirements)
    workflow.add_node("map_evidence", map_evidence)
    workflow.add_node("score_coverage", score_coverage)
    workflow.add_node("check_unsupported_claims", check_unsupported_claims)
    workflow.add_node("human_approval_gate", human_approval_gate)
    workflow.add_node("draft_final_response", draft_final_response)

    workflow.add_edge(START, "load_inputs")
    workflow.add_edge("load_inputs", "extract_requirements")
    workflow.add_edge("extract_requirements", "map_evidence")
    workflow.add_edge("map_evidence", "score_coverage")
    workflow.add_edge("score_coverage", "check_unsupported_claims")
    workflow.add_edge("check_unsupported_claims", "human_approval_gate")
    workflow.add_conditional_edges(
        "human_approval_gate",
        route_after_human_gate,
        {
            "draft_final_response": "draft_final_response",
            "end": END,
        },
    )
    workflow.add_edge("draft_final_response", END)

    return workflow.compile()
