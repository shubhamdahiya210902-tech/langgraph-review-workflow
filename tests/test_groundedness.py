from langgraph_review_workflow.steps.groundedness import check_claim_groundedness


def test_groundedness_blocks_unsupported_and_contradicted_claims(
    candidate_profile: dict,
) -> None:
    result = check_claim_groundedness({"candidate_profile": candidate_profile})
    checks_by_id = {check.claim_id: check for check in result.checks}

    assert result.grounded_count == 3
    assert result.partial_count == 0
    assert result.unsupported_count == 3
    assert result.contradicted_count == 1
    assert len(result.unsupported_claims) == 4

    assert checks_by_id["CLAIM-001"].status == "grounded"
    assert checks_by_id["CLAIM-004"].status == "unsupported"
    assert checks_by_id["CLAIM-005"].status == "unsupported"
    assert checks_by_id["CLAIM-006"].status == "contradicted"
    assert checks_by_id["CLAIM-006"].severity == "high"
    assert checks_by_id["CLAIM-007"].status == "unsupported"


def test_groundedness_does_not_polish_blocked_claims(candidate_profile: dict) -> None:
    result = check_claim_groundedness({"candidate_profile": candidate_profile})

    for claim in result.unsupported_claims:
        assert claim.safe_action.startswith("Block")
        assert claim.status != "grounded"
