"""Live-evaluation tests.

The fake-client tests run in normal CI and check the live wiring and metric math
without a model. The `live` test calls a local Ollama model and is skipped
unless RUN_LIVE=1 and Ollama is reachable.
"""

import os
from typing import Any

import pytest
from pydantic import BaseModel

from langgraph_review_workflow import live_eval
from langgraph_review_workflow.graph import build_graph
from langgraph_review_workflow.live_eval import (
    JudgeCriterion,
    JudgeVerdict,
    LiveRunRecord,
    block_decision_agreement,
    claim_metrics,
    cosine_similarity,
    rule_check_draft,
    run_live_eval,
)
from langgraph_review_workflow.model_boundary import ollama_model_boundary
from langgraph_review_workflow.schemas import (
    LiveClaimLabel,
    LiveDraftOutput,
    LiveGroundednessLabels,
)
from langgraph_review_workflow.steps import live_drafting, live_groundedness
from langgraph_review_workflow.steps.live_groundedness import labels_to_result


EXPECTED_LABELS = {
    "CLAIM-001": "grounded",
    "CLAIM-002": "grounded",
    "CLAIM-003": "grounded",
    "CLAIM-004": "unsupported",
    "CLAIM-005": "unsupported",
    "CLAIM-006": "contradicted",
    "CLAIM-007": "unsupported",
}
EVIDENCE_FOR = {"CLAIM-001": ["EV-001"], "CLAIM-002": ["EV-002"], "CLAIM-003": ["EV-003"]}


class FakeChatClient:
    """Returns canned structured outputs keyed by the requested schema."""

    def __init__(self, labels: dict[str, str], draft_text: str = "Safe draft. EV-001") -> None:
        self.labels = labels
        self.draft_text = draft_text

    @classmethod
    def from_boundary(cls, boundary: Any) -> "FakeChatClient":
        return cls(EXPECTED_LABELS)

    def structured_chat(self, *, schema: type[BaseModel], **_: Any) -> BaseModel:
        if schema is LiveGroundednessLabels:
            return LiveGroundednessLabels(
                labels=[
                    LiveClaimLabel(
                        claim_id=claim_id,
                        status=status,
                        evidence_ids=EVIDENCE_FOR.get(claim_id, []),
                        rationale="fake",
                    )
                    for claim_id, status in self.labels.items()
                ]
            )
        if schema is LiveDraftOutput:
            return LiveDraftOutput(draft_text=self.draft_text, cited_evidence_ids=["EV-001"])
        if schema is JudgeVerdict:
            criterion = JudgeCriterion(score=5, reason="fake")
            return JudgeVerdict(
                only_grounded_claims=criterion,
                no_blocked_claims=criterion,
                no_new_facts=criterion,
            )
        raise AssertionError(f"unexpected schema {schema}")


class FakeEmbedClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_missing_live_label_is_blocked_and_unknown_evidence_dropped(
    candidate_profile: dict,
) -> None:
    labels = LiveGroundednessLabels(
        labels=[
            LiveClaimLabel(
                claim_id="claim-001",
                status="grounded",
                evidence_ids=["EV-001", "EV-999"],
                rationale="Supported by the RAG project.",
            )
        ]
    )

    result = labels_to_result(labels, candidate_profile)
    checks = {check.claim_id: check for check in result.checks}

    assert len(result.checks) == 7
    assert checks["CLAIM-001"].status == "grounded"
    assert [e.evidence_id for e in checks["CLAIM-001"].evidence] == ["EV-001"]
    assert checks["CLAIM-002"].status == "unsupported"
    assert result.grounded_count == 1
    assert len(result.unsupported_claims) == 6


def test_graph_uses_live_steps_only_for_ollama_boundary(
    candidate_profile: dict,
    job_description: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_groundedness, "OllamaClient", FakeChatClient)
    monkeypatch.setattr(live_drafting, "OllamaClient", FakeChatClient)

    result = build_graph().invoke(
        {
            "candidate_profile": candidate_profile,
            "job_description": job_description,
            "model_boundary": ollama_model_boundary().model_dump(mode="json"),
            "human_approved": True,
            "audit_trail": [],
        }
    )

    assert result["groundedness_summary"]["summary"].startswith("Live model checked 7")
    assert result["approval_decision"]["approved_claim_ids"] == [
        "CLAIM-001",
        "CLAIM-002",
        "CLAIM-003",
    ]
    assert result["final_draft"] == "Safe draft. EV-001"


def test_agreement_and_consistency_metrics() -> None:
    expected = {"CLAIM-001": "grounded", "CLAIM-006": "contradicted"}
    runs = [
        LiveRunRecord(run_index=1, duration_seconds=1, labels=dict(expected)),
        LiveRunRecord(
            run_index=2,
            duration_seconds=1,
            labels={"CLAIM-001": "grounded", "CLAIM-006": "unsupported"},
        ),
        LiveRunRecord(run_index=3, duration_seconds=1, error="boom"),
        LiveRunRecord(run_index=4, duration_seconds=1, labels=dict(expected)),
    ]

    metrics = {m.claim_id: m for m in claim_metrics(expected, {}, runs)}

    assert metrics["CLAIM-001"].agreement == 0.75
    assert metrics["CLAIM-006"].agreement == 0.5
    assert metrics["CLAIM-006"].consistency == 0.5
    assert metrics["CLAIM-006"].changed_label is True
    # unsupported vs contradicted is still a correct block decision; the failed run is not.
    assert block_decision_agreement(expected, runs) == 6 / 8


def test_rule_checks_flag_pronouns_denials_and_blocked_assertions() -> None:
    grounded = {"CLAIM-003": "Compared synthetic LLM responses with a factuality rubric."}
    blocked = {"CLAIM-006": "Expert-level German fluency."}

    clean = rule_check_draft(
        "The candidate compared synthetic LLM responses with a factuality rubric (EV-003). "
        "There is no evidence of expert-level German fluency.",
        grounded,
        blocked,
    )
    flawed = rule_check_draft(
        "She did not compare synthetic responses with a factuality rubric. "
        "She has expert-level German fluency.",
        grounded,
        blocked,
    )

    assert clean.passed
    assert flawed.gendered_pronoun
    assert flawed.denied_grounded_claim_ids == ["CLAIM-003"]
    assert flawed.asserted_blocked_claim_ids == ["CLAIM-006"]
    assert not flawed.passed


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_run_live_eval_with_fake_clients(
    candidate_profile: dict,
    job_description: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_groundedness, "OllamaClient", FakeChatClient)
    monkeypatch.setattr(live_drafting, "OllamaClient", FakeChatClient)

    report = run_live_eval(
        candidate_profile=candidate_profile,
        job_description=job_description,
        runs=3,
        judge_client=FakeChatClient(EXPECTED_LABELS),
        embed_client=FakeEmbedClient(),  # type: ignore[arg-type]
    )

    assert report.runs_completed == 3
    assert report.overall_label_agreement == 1.0
    assert report.claims_that_changed_label == 0
    assert report.runs_approving_blocked_claim == 0
    assert report.mean_semantic_similarity == 1.0
    assert report.judge_pass_rate == 1.0
    assert report.rule_check_pass_rate == 1.0
    assert report.judge_rule_agreement == 1.0
    assert "| Label agreement" in live_eval.render_markdown(report)


def test_write_report_keeps_earlier_results(
    candidate_profile: dict,
    job_description: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(live_groundedness, "OllamaClient", FakeChatClient)
    monkeypatch.setattr(live_drafting, "OllamaClient", FakeChatClient)
    report = run_live_eval(
        candidate_profile=candidate_profile,
        job_description=job_description,
        runs=1,
        judge_client=FakeChatClient(EXPECTED_LABELS),
        embed_client=FakeEmbedClient(),  # type: ignore[arg-type]
    )

    first, _ = live_eval.write_report(report, tmp_path)
    second, _ = live_eval.write_report(report, tmp_path)

    assert first != second
    assert first.exists() and second.exists()


@pytest.mark.live
def test_live_model_meets_regression_thresholds(
    candidate_profile: dict,
    job_description: str,
    tmp_path: Any,
) -> None:
    """Non-regression gate for the live model; thresholds are set via env vars.

    Defaults sit just below the measured runs (eval/results/live_eval_2026-11-01-2
    and -3: 70.0% label agreement, 95.7-100% allow/block agreement, 40% judge pass
    rate with the llama3.2:3b judge) so a prompt or model change that regresses fails.
    The judge threshold guards against regression; it is not a quality bar.
    """

    agreement_threshold = float(os.environ.get("LIVE_AGREEMENT_THRESHOLD", "0.6"))
    block_threshold = float(os.environ.get("LIVE_BLOCK_AGREEMENT_THRESHOLD", "0.9"))
    judge_threshold = float(os.environ.get("LIVE_JUDGE_THRESHOLD", "0.3"))

    report = run_live_eval(
        candidate_profile=candidate_profile,
        job_description=job_description,
        runs=int(os.environ.get("LIVE_EVAL_RUNS", "10")),
        temperature=float(os.environ.get("LIVE_EVAL_TEMPERATURE", "0.7")),
        model=os.environ.get("LIVE_EVAL_MODEL", live_eval.DEFAULT_OLLAMA_MODEL),
        judge_model=os.environ.get("LIVE_JUDGE_MODEL", live_eval.DEFAULT_OLLAMA_JUDGE_MODEL),
        base_url=os.environ.get("OLLAMA_BASE_URL", live_eval.DEFAULT_OLLAMA_BASE_URL),
    )
    live_eval.write_report(report, tmp_path)

    assert report.runs_failed == 0, [r.error for r in report.runs if r.error]
    assert report.overall_label_agreement >= agreement_threshold
    assert report.block_decision_agreement >= block_threshold
    assert report.judge_pass_rate is not None
    assert report.judge_pass_rate >= judge_threshold
