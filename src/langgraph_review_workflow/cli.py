"""Command-line entrypoint for the review workflow demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from langgraph_review_workflow.graph import build_graph
from langgraph_review_workflow.model_boundary import DEFAULT_MODEL_BOUNDARY
from langgraph_review_workflow.state import ReviewState


DEFAULT_CANDIDATE_PROFILE = Path("data/mock/candidate_profile.json")
DEFAULT_JOB_DESCRIPTION = Path("data/mock/job_description.txt")

app = typer.Typer(
    help="Run the public-safe LangGraph review workflow demo.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Public-safe review workflow CLI."""


def _load_candidate_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_job_description(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@app.command()
def demo(
    candidate_profile: Path = typer.Option(
        DEFAULT_CANDIDATE_PROFILE,
        "--candidate-profile",
        help="Path to a synthetic candidate profile JSON file.",
    ),
    job_description: Path = typer.Option(
        DEFAULT_JOB_DESCRIPTION,
        "--job-description",
        help="Path to a synthetic job description text file.",
    ),
    approve_final: bool = typer.Option(
        False,
        "--approve-final",
        help="Simulate human approval for safe-only drafting with grounded claims.",
    ),
    details: bool = typer.Option(
        False,
        "--details",
        help="Show detailed extraction, evidence, and scorecard tables.",
    ),
) -> None:
    """Run the public-safe demo workflow against mock inputs."""

    initial_state: ReviewState = {
        "candidate_profile_path": str(candidate_profile),
        "candidate_profile": _load_candidate_profile(candidate_profile),
        "job_description_path": str(job_description),
        "job_description": _load_job_description(job_description),
        "model_boundary": DEFAULT_MODEL_BOUNDARY.model_dump(mode="json"),
        "human_approved": approve_final,
        "audit_trail": [],
    }

    result = build_graph().invoke(initial_state)

    table = Table(title="LangGraph Review Workflow Demo")
    table.add_column("Node", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Message")

    for event in result.get("audit_trail", []):
        table.add_row(event["node"], event["status"], event["message"])

    console.print(table)

    requirements = result.get("requirements", [])
    if details and requirements:
        requirements_table = Table(title="Extracted Requirements")
        requirements_table.add_column("ID", style="cyan", no_wrap=True)
        requirements_table.add_column("Priority", style="magenta")
        requirements_table.add_column("Category", style="green")
        requirements_table.add_column("Source")
        requirements_table.add_column("Requirement")

        for requirement in requirements:
            requirements_table.add_row(
                requirement["requirement_id"],
                requirement["priority"],
                requirement["category"],
                requirement["source_section"],
                requirement["text"],
            )

        console.print(requirements_table)

    evidence_map = result.get("evidence_map", [])
    if details and evidence_map:
        evidence_table = Table(title="Evidence Mapping")
        evidence_table.add_column("Req", style="cyan", no_wrap=True)
        evidence_table.add_column("Support", style="magenta")
        evidence_table.add_column("Evidence IDs", style="green")
        evidence_table.add_column("Matched Concepts")
        evidence_table.add_column("Rationale")

        for mapping in evidence_map:
            evidence_ids = ", ".join(
                evidence["evidence_id"] for evidence in mapping.get("evidence", [])
            )
            matched_concepts = ", ".join(mapping.get("matched_concepts", []))
            evidence_table.add_row(
                mapping["requirement_id"],
                mapping["support_level"],
                evidence_ids or "-",
                matched_concepts or "-",
                mapping["rationale"],
            )

        console.print(evidence_table)

    coverage_summary = result.get("coverage_summary")
    coverage_scores = result.get("coverage_scores", [])
    if coverage_summary and coverage_scores:
        console.print(
            "[bold]Coverage summary:[/bold] "
            f"{coverage_summary['summary']} "
            f"Gaps: {', '.join(coverage_summary['gap_requirement_ids']) or '-'}."
        )

        if details:
            score_table = Table(title="Coverage Scorecard")
            score_table.add_column("Req", style="cyan", no_wrap=True)
            score_table.add_column("Priority", style="magenta")
            score_table.add_column("Status", style="green")
            score_table.add_column("Score", justify="right")
            score_table.add_column("Evidence IDs")
            score_table.add_column("Gap Summary")

            for score in coverage_scores:
                evidence_ids = ", ".join(score.get("evidence_ids", []))
                score_table.add_row(
                    score["requirement_id"],
                    score["priority"],
                    score["coverage_status"],
                    f"{score['score']:.1f}/{score['max_score']:.1f}",
                    evidence_ids or "-",
                    score["gap_summary"],
                )

            console.print(score_table)

    groundedness_summary = result.get("groundedness_summary")
    groundedness_checks = result.get("groundedness_checks", [])
    if groundedness_summary and groundedness_checks:
        console.print(f"[bold]Groundedness summary:[/bold] {groundedness_summary['summary']}")

        if details:
            claims_table = Table(title="Groundedness Checks")
            claims_table.add_column("ID", style="cyan", no_wrap=True)
            claims_table.add_column("Status", style="magenta")
            claims_table.add_column("Severity", style="red")
            claims_table.add_column("Evidence IDs", style="green")
            claims_table.add_column("Claim")
            claims_table.add_column("Safe Action")

            for check in groundedness_checks:
                evidence_ids = ", ".join(
                    evidence["evidence_id"] for evidence in check.get("evidence", [])
                )
                claims_table.add_row(
                    check["claim_id"],
                    check["status"],
                    check["severity"],
                    evidence_ids or "-",
                    check["claim"],
                    check["safe_action"],
                )

            console.print(claims_table)

        unsafe_claims = result.get("unsupported_claims", [])
        if unsafe_claims:
            unsafe_table = Table(title="Claims Blocked From Drafting")
            unsafe_table.add_column("ID", style="cyan", no_wrap=True)
            unsafe_table.add_column("Status", style="magenta")
            unsafe_table.add_column("Claim")
            unsafe_table.add_column("Reason")

            for claim in unsafe_claims:
                unsafe_table.add_row(
                    claim["claim_id"],
                    claim["status"],
                    claim["claim"],
                    claim["explanation"],
                )

            console.print(unsafe_table)

    approval_decision = result.get("approval_decision")
    if approval_decision:
        console.print(f"[bold]Approval gate:[/bold] {approval_decision['summary']}")

        approval_table = Table(title="Human Approval Decision")
        approval_table.add_column("Status", style="magenta")
        approval_table.add_column("Can Draft", style="green")
        approval_table.add_column("Approved Claims")
        approval_table.add_column("Blocked Claims")
        approval_table.add_column("Required Action")

        approval_table.add_row(
            approval_decision["approval_status"],
            str(approval_decision["can_draft"]),
            ", ".join(approval_decision.get("approved_claim_ids", [])) or "-",
            ", ".join(approval_decision.get("blocked_claim_ids", [])) or "-",
            approval_decision["required_human_action"],
        )
        console.print(approval_table)

    console.print(f"Approval status: [bold]{result.get('approval_status', 'pending')}[/bold]")
    if result.get("final_draft"):
        console.print(Markdown(result["final_draft"]))
    else:
        console.print("Final draft: [dim]skipped until human approval is provided[/dim]")


if __name__ == "__main__":
    app()
