from langgraph_review_workflow.steps.scoring import score_coverage_from_evidence_map


def test_coverage_scoring_is_weighted_and_explainable(evidence_map: list[dict]) -> None:
    result = score_coverage_from_evidence_map({"evidence_map": evidence_map})
    scores_by_id = {score.requirement_id: score for score in result.requirement_scores}

    assert result.total_score == 31.0
    assert result.max_score == 34.0
    assert result.coverage_percent == 91.2
    assert result.gap_requirement_ids == ["REQ-012", "REQ-015"]
    assert result.partial_requirement_ids == ["REQ-001"]

    assert scores_by_id["REQ-001"].coverage_status == "partial"
    assert scores_by_id["REQ-001"].score == 1.0
    assert scores_by_id["REQ-012"].coverage_status == "gap"
    assert scores_by_id["REQ-017"].coverage_status == "excluded_policy"
    assert scores_by_id["REQ-017"].max_score == 0.0
