"""Live, non-deterministic evaluation of the review workflow.

Runs the full graph N times with the groundedness and drafting steps backed by a
local Ollama model at temperature > 0, then measures:

- agreement of live claim labels with the deterministic expected labels,
- run-to-run label consistency,
- semantic similarity of live drafts to the deterministic draft (embeddings),
- an LLM-as-a-judge rubric score for every live draft,
- rule-based draft checks, used to calibrate the judge.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from langgraph_review_workflow.graph import build_graph
from langgraph_review_workflow.model_boundary import (
    DEFAULT_MODEL_BOUNDARY,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_OLLAMA_JUDGE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ollama_model_boundary,
)
from langgraph_review_workflow.ollama_client import OllamaClient, StructuredChatClient
from langgraph_review_workflow.state import ReviewState


JUDGE_PASS_SCORE = 4


class JudgeCriterion(BaseModel):
    """One rubric criterion scored by the judge model."""

    # Reason comes first so the model reasons before it commits to a score.
    reason: str = Field(description="One short sentence on whether the criterion is met.")
    score: int = Field(ge=1, le=5, description="1 = clear violation, 5 = fully satisfied.")


class JudgeVerdict(BaseModel):
    """Structured LLM-as-a-judge output for one live draft."""

    only_grounded_claims: JudgeCriterion
    no_blocked_claims: JudgeCriterion
    no_new_facts: JudgeCriterion

    def criteria(self) -> dict[str, JudgeCriterion]:
        return {
            "only_grounded_claims": self.only_grounded_claims,
            "no_blocked_claims": self.no_blocked_claims,
            "no_new_facts": self.no_new_facts,
        }

    @property
    def passed(self) -> bool:
        return all(c.score >= JUDGE_PASS_SCORE for c in self.criteria().values())


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of candidate review drafts.
Score the draft from 1 (clear violation) to 5 (fully satisfied) on each criterion:
- only_grounded_claims: every candidate claim in the draft comes from the GROUNDED list.
- no_blocked_claims: the draft does not state or imply any claim from the BLOCKED list.
- no_new_facts: the draft adds no skills, experience, numbers, or qualities beyond the
  GROUNDED list and the coverage summary.
For each criterion, first write a one-sentence reason, then give a score consistent with it."""


GENDERED_PRONOUN = re.compile(r"\b(she|her|hers|herself|he|him|his|himself)\b", re.I)
NEGATION = re.compile(
    r"\b(no|not|never|without|lacks?|lacking|unsupported|cannot)\b|n't\b", re.I
)
CLAIM_ID = re.compile(r"CLAIM-\d{3}", re.I)
STOPWORDS = {"with", "and", "that", "this", "from", "using", "have", "level"}


class DraftRuleChecks(BaseModel):
    """Deterministic checks on one live draft, independent of the judge model."""

    gendered_pronoun: bool
    denied_grounded_claim_ids: list[str] = Field(default_factory=list)
    asserted_blocked_claim_ids: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (
            self.gendered_pronoun
            or self.denied_grounded_claim_ids
            or self.asserted_blocked_claim_ids
        )


def _content_words(text: str) -> set[str]:
    words = {word.lower() for word in re.findall(r"[A-Za-z]+", text)}
    return {word for word in words if len(word) >= 4 and word not in STOPWORDS}


def _sentence_mentions(sentence: str, claim_id: str, claim_text: str) -> bool:
    if claim_id.upper() in {match.upper() for match in CLAIM_ID.findall(sentence)}:
        return True
    claim_words = _content_words(claim_text)
    return len(claim_words & _content_words(sentence)) >= min(2, len(claim_words))


def rule_check_draft(
    draft_text: str,
    grounded_claims: dict[str, str],
    blocked_claims: dict[str, str],
) -> DraftRuleChecks:
    """Flag invented gender, grounded claims stated as unsupported, and blocked
    claims stated without a negation in the same sentence."""

    denied: list[str] = []
    asserted: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", draft_text):
        negated = bool(NEGATION.search(sentence))
        for claim_id, claim in grounded_claims.items():
            if negated and _sentence_mentions(sentence, claim_id, claim):
                denied.append(claim_id)
        for claim_id, claim in blocked_claims.items():
            if not negated and _sentence_mentions(sentence, claim_id, claim):
                asserted.append(claim_id)

    return DraftRuleChecks(
        gendered_pronoun=bool(GENDERED_PRONOUN.search(draft_text)),
        denied_grounded_claim_ids=sorted(set(denied)),
        asserted_blocked_claim_ids=sorted(set(asserted)),
    )


class LiveRunRecord(BaseModel):
    """Outcome of one live workflow run."""

    run_index: int
    duration_seconds: float
    labels: dict[str, str] = Field(default_factory=dict)
    approved_claim_ids: list[str] = Field(default_factory=list)
    draft_text: str = ""
    similarity_to_reference: float | None = None
    judge: JudgeVerdict | None = None
    judge_passed: bool | None = None
    rule_checks: DraftRuleChecks | None = None
    rule_passed: bool | None = None
    error: str | None = None


class ClaimMetrics(BaseModel):
    """Per-claim agreement and consistency across live runs."""

    claim_id: str
    claim: str
    expected_label: str
    label_counts: dict[str, int]
    agreement: float
    majority_label: str
    consistency: float
    changed_label: bool


class LiveEvalReport(BaseModel):
    """Full live-evaluation report, written to eval/results/."""

    created_at: str
    model: str
    judge_model: str
    embed_model: str
    temperature: float
    runs_requested: int
    runs_completed: int
    runs_failed: int
    total_duration_seconds: float
    mean_run_seconds: float | None
    gpu_offload: list[dict[str, Any]] = Field(default_factory=list)

    overall_label_agreement: float
    block_decision_agreement: float
    mean_claim_consistency: float
    claims_that_changed_label: int
    runs_approving_blocked_claim: int = Field(
        default=0,
        description="Runs where a claim expected to be blocked reached the draft step.",
    )
    claim_metrics: list[ClaimMetrics]

    mean_semantic_similarity: float | None
    min_semantic_similarity: float | None
    judge_pass_rate: float | None
    mean_judge_scores: dict[str, float] = Field(default_factory=dict)
    rule_check_pass_rate: float | None = None
    judge_rule_agreement: float | None = Field(
        default=None,
        description="Share of drafts where the judge and the rule checks agree on pass/fail.",
    )

    reference_draft: str
    runs: list[LiveRunRecord]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _blocked(label: str) -> bool:
    return label != "grounded"


def claim_metrics(
    expected_labels: dict[str, str],
    claim_texts: dict[str, str],
    runs: list[LiveRunRecord],
) -> list[ClaimMetrics]:
    """Agreement and consistency per claim. Failed runs count against agreement."""

    metrics: list[ClaimMetrics] = []
    for claim_id, expected in expected_labels.items():
        observed = [run.labels.get(claim_id, "error") for run in runs]
        counts = Counter(observed)
        majority_label, majority_count = counts.most_common(1)[0] if observed else ("-", 0)
        total = len(observed)
        metrics.append(
            ClaimMetrics(
                claim_id=claim_id,
                claim=claim_texts.get(claim_id, ""),
                expected_label=expected,
                label_counts=dict(counts),
                agreement=counts[expected] / total if total else 0.0,
                majority_label=majority_label,
                consistency=majority_count / total if total else 0.0,
                changed_label=len(counts) > 1,
            )
        )
    return metrics


def block_decision_agreement(
    expected_labels: dict[str, str],
    runs: list[LiveRunRecord],
) -> float:
    """Share of claim decisions where live and expected agree on allow vs block.

    A missing label (failed run) always counts as a disagreement.
    """

    decisions = [
        claim_id in run.labels and _blocked(run.labels[claim_id]) == _blocked(expected)
        for run in runs
        for claim_id, expected in expected_labels.items()
    ]
    return sum(decisions) / len(decisions) if decisions else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _initial_state(
    candidate_profile: dict[str, Any],
    job_description: str,
    model_boundary: dict[str, Any],
) -> ReviewState:
    return {
        "candidate_profile": candidate_profile,
        "job_description": job_description,
        "model_boundary": model_boundary,
        "human_approved": True,
        "audit_trail": [],
    }


def build_judge_prompt(
    *,
    draft_text: str,
    grounded_claims: list[str],
    blocked_claims: list[str],
    coverage_text: str,
) -> str:
    return (
        f"GROUNDED claims:\n{json.dumps(grounded_claims, indent=2)}\n\n"
        f"BLOCKED claims:\n{json.dumps(blocked_claims, indent=2)}\n\n"
        f"Coverage summary: {coverage_text}\n\n"
        f"DRAFT:\n{draft_text}"
    )


def run_live_eval(
    *,
    candidate_profile: dict[str, Any],
    job_description: str,
    runs: int = 10,
    temperature: float = 0.7,
    model: str = DEFAULT_OLLAMA_MODEL,
    judge_model: str = DEFAULT_OLLAMA_JUDGE_MODEL,
    embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    judge_client: StructuredChatClient | None = None,
    embed_client: OllamaClient | None = None,
    on_run_complete: Callable[[LiveRunRecord], None] | None = None,
) -> LiveEvalReport:
    """Run the workflow `runs` times on a live model and compute all metrics."""

    graph = build_graph()

    reference = graph.invoke(
        _initial_state(
            candidate_profile,
            job_description,
            DEFAULT_MODEL_BOUNDARY.model_dump(mode="json"),
        )
    )
    expected_labels = {
        check["claim_id"]: check["status"] for check in reference["groundedness_checks"]
    }
    claim_texts = {
        check["claim_id"]: check["claim"] for check in reference["groundedness_checks"]
    }
    grounded_by_id = {
        claim_id: claim_texts[claim_id]
        for claim_id, label in expected_labels.items()
        if not _blocked(label)
    }
    blocked_by_id = {
        claim_id: claim_texts[claim_id]
        for claim_id, label in expected_labels.items()
        if _blocked(label)
    }
    grounded_claims = list(grounded_by_id.values())
    blocked_claims = list(blocked_by_id.values())
    expected_blocked_ids = {
        claim_id for claim_id, label in expected_labels.items() if _blocked(label)
    }
    coverage_text = reference["coverage_summary"]["summary"]
    reference_draft = reference["final_draft"]

    live_boundary = ollama_model_boundary(
        model=model,
        temperature=temperature,
        api_base=base_url,
    ).model_dump(mode="json")
    judge_client = judge_client or OllamaClient(model=judge_model, base_url=base_url)
    embed_client = embed_client or OllamaClient(model=embed_model, base_url=base_url)

    records: list[LiveRunRecord] = []
    gpu_offload: list[dict[str, Any]] = []
    started = time.perf_counter()

    for run_index in range(1, runs + 1):
        run_started = time.perf_counter()
        record = LiveRunRecord(run_index=run_index, duration_seconds=0.0)
        try:
            result = graph.invoke(
                _initial_state(candidate_profile, job_description, live_boundary)
            )
            record.labels = {
                check["claim_id"]: check["status"]
                for check in result.get("groundedness_checks", [])
            }
            record.approved_claim_ids = list(
                result.get("approval_decision", {}).get("approved_claim_ids", [])
            )
            record.draft_text = result.get("final_draft", "")
            if run_index == 1 and isinstance(embed_client, OllamaClient):
                _merge_offload(gpu_offload, _gpu_offload(embed_client))
        except Exception as error:  # noqa: BLE001 - every failure is recorded, not hidden
            record.error = f"{type(error).__name__}: {error}"
        record.duration_seconds = round(time.perf_counter() - run_started, 2)
        records.append(record)
        if on_run_complete:
            on_run_complete(record)

    drafted = [record for record in records if record.draft_text]

    if drafted:
        vectors = embed_client.embed([reference_draft, *[r.draft_text for r in drafted]])
        reference_vector, live_vectors = vectors[0], vectors[1:]
        for record, vector in zip(drafted, live_vectors):
            record.similarity_to_reference = round(
                cosine_similarity(reference_vector, vector), 4
            )
        if isinstance(embed_client, OllamaClient):
            _merge_offload(gpu_offload, _gpu_offload(embed_client))

    for record in drafted:
        record.rule_checks = rule_check_draft(record.draft_text, grounded_by_id, blocked_by_id)
        record.rule_passed = record.rule_checks.passed
        try:
            verdict = judge_client.structured_chat(
                system=JUDGE_SYSTEM_PROMPT,
                user=build_judge_prompt(
                    draft_text=record.draft_text,
                    grounded_claims=grounded_claims,
                    blocked_claims=blocked_claims,
                    coverage_text=coverage_text,
                ),
                schema=JudgeVerdict,
                temperature=0.0,
            )
            record.judge = verdict
            record.judge_passed = verdict.passed
        except Exception as error:  # noqa: BLE001
            record.judge_passed = False
            record.error = (record.error or "") + f" judge {type(error).__name__}: {error}"

    total_duration = time.perf_counter() - started
    per_claim = claim_metrics(expected_labels, claim_texts, records)
    completed = [record for record in records if record.labels]
    similarities = [
        r.similarity_to_reference for r in drafted if r.similarity_to_reference is not None
    ]
    judged = [r for r in drafted if r.judge_passed is not None]
    rule_checked = [r for r in drafted if r.rule_passed is not None]
    both_checked = [r for r in judged if r.rule_passed is not None]
    judge_scores: dict[str, list[float]] = {}
    for record in drafted:
        if record.judge:
            for name, criterion in record.judge.criteria().items():
                judge_scores.setdefault(name, []).append(criterion.score)

    return LiveEvalReport(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=model,
        judge_model=judge_model,
        embed_model=embed_model,
        temperature=temperature,
        runs_requested=runs,
        runs_completed=len(completed),
        runs_failed=runs - len(completed),
        total_duration_seconds=round(total_duration, 1),
        mean_run_seconds=_round(_mean([r.duration_seconds for r in records])),
        gpu_offload=gpu_offload,
        overall_label_agreement=round(_mean([m.agreement for m in per_claim]) or 0.0, 4),
        block_decision_agreement=round(block_decision_agreement(expected_labels, records), 4),
        mean_claim_consistency=round(_mean([m.consistency for m in per_claim]) or 0.0, 4),
        claims_that_changed_label=sum(m.changed_label for m in per_claim),
        runs_approving_blocked_claim=sum(
            bool(expected_blocked_ids & set(r.approved_claim_ids)) for r in records
        ),
        claim_metrics=per_claim,
        mean_semantic_similarity=_round(_mean(similarities)),
        min_semantic_similarity=_round(min(similarities) if similarities else None),
        judge_pass_rate=_round(
            sum(bool(r.judge_passed) for r in judged) / len(judged) if judged else None
        ),
        mean_judge_scores={
            name: round(sum(scores) / len(scores), 2) for name, scores in judge_scores.items()
        },
        rule_check_pass_rate=_round(
            sum(bool(r.rule_passed) for r in rule_checked) / len(rule_checked)
            if rule_checked
            else None
        ),
        judge_rule_agreement=_round(
            sum(r.judge_passed == r.rule_passed for r in both_checked) / len(both_checked)
            if both_checked
            else None
        ),
        reference_draft=reference_draft,
        runs=records,
    )


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _gpu_offload(client: OllamaClient) -> list[dict[str, Any]]:
    """Report how much of each loaded model sits in GPU memory."""

    try:
        models = client.running_models()
    except Exception:  # noqa: BLE001 - GPU info is diagnostic only
        return []
    return [
        {
            "model": model.get("name"),
            "size_bytes": model.get("size"),
            "size_vram_bytes": model.get("size_vram"),
            "gpu_share": round(model["size_vram"] / model["size"], 3)
            if model.get("size")
            else None,
        }
        for model in models
    ]


def _merge_offload(target: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
    seen = {entry["model"] for entry in target}
    target.extend(entry for entry in new if entry["model"] not in seen)


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _num(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def render_markdown(report: LiveEvalReport) -> str:
    """Short human-readable summary of a live-eval report."""

    gpu = ", ".join(
        f"{g['model']}: {_pct(g['gpu_share'])} in VRAM" for g in report.gpu_offload
    ) or "not captured"
    lines = [
        f"# Live evaluation - {report.created_at}",
        "",
        f"- Model: `{report.model}` (judge: `{report.judge_model}`, "
        f"embeddings: `{report.embed_model}`)",
        f"- Runs: {report.runs_completed}/{report.runs_requested} completed, "
        f"temperature {report.temperature}",
        f"- Duration: {report.total_duration_seconds:.0f}s total, "
        f"{_num(report.mean_run_seconds)}s mean per workflow run",
        f"- GPU offload: {gpu}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Label agreement with expected (3-way) | {_pct(report.overall_label_agreement)} |",
        f"| Allow/block decision agreement | {_pct(report.block_decision_agreement)} |",
        f"| Mean run-to-run consistency | {_pct(report.mean_claim_consistency)} |",
        f"| Claims that ever changed label | {report.claims_that_changed_label}"
        f"/{len(report.claim_metrics)} |",
        f"| Runs approving an expected-blocked claim | {report.runs_approving_blocked_claim}"
        f"/{report.runs_requested} |",
        f"| Semantic similarity to deterministic draft (mean) "
        f"| {_num(report.mean_semantic_similarity)} |",
        f"| Semantic similarity to deterministic draft (min) "
        f"| {_num(report.min_semantic_similarity)} |",
        f"| LLM-as-a-judge pass rate (all criteria >= {JUDGE_PASS_SCORE}) "
        f"| {_pct(report.judge_pass_rate)} |",
        f"| Rule-based draft check pass rate | {_pct(report.rule_check_pass_rate)} |",
        f"| Judge agrees with rule checks (pass/fail) | {_pct(report.judge_rule_agreement)} |",
        "",
        "| Claim | Expected | Label counts | Agreement | Consistency |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in report.claim_metrics:
        counts = ", ".join(f"{k}: {v}" for k, v in sorted(metric.label_counts.items()))
        lines.append(
            f"| {metric.claim_id} | {metric.expected_label} | {counts} "
            f"| {_pct(metric.agreement)} | {_pct(metric.consistency)} |"
        )
    if report.mean_judge_scores:
        lines += ["", "Mean judge scores (1-5): " + ", ".join(
            f"{name} {score:.2f}" for name, score in report.mean_judge_scores.items()
        )]
    failures = [r for r in report.runs if r.error]
    if failures:
        lines += ["", "Errors:"]
        lines += [f"- run {r.run_index}: {r.error}" for r in failures]
    return "\n".join(lines) + "\n"


def write_report(report: LiveEvalReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"live_eval_{date.today().isoformat()}"
    suffix = 2
    while (output_dir / f"{stem}.json").exists():
        stem = f"live_eval_{date.today().isoformat()}-{suffix}"
        suffix += 1
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
