"""Structured output schemas used by workflow steps."""

from typing import Literal

from pydantic import BaseModel, Field


RequirementCategory = Literal[
    "responsibility",
    "required_qualification",
    "nice_to_have",
    "language",
    "claim_policy",
]

RequirementPriority = Literal["must_have", "should_have", "nice_to_have", "policy"]
SupportLevel = Literal["supported", "partially_supported", "not_supported", "policy_constraint"]
CoverageStatus = Literal["full", "partial", "gap", "excluded_policy"]
ClaimStatus = Literal["grounded", "partially_grounded", "unsupported", "contradicted"]
ClaimSeverity = Literal["none", "low", "medium", "high"]
ApprovalDecisionStatus = Literal["pending", "approved", "rejected"]


class ExtractedRequirement(BaseModel):
    """Single job requirement or review constraint extracted from a job description."""

    requirement_id: str = Field(description="Stable requirement identifier, such as REQ-001.")
    category: RequirementCategory
    priority: RequirementPriority
    text: str = Field(description="Requirement text copied from the job description.")
    source_section: str = Field(description="Job-description section where the item appeared.")
    keywords: list[str] = Field(
        default_factory=list,
        description="Simple matching keywords derived from the requirement text.",
    )


class RequirementExtractionResult(BaseModel):
    """Structured output for the requirement extraction step."""

    role_title: str
    company: str | None = None
    requirements: list[ExtractedRequirement] = Field(default_factory=list)


class EvidenceMatch(BaseModel):
    """Evidence item that supports part or all of a requirement."""

    evidence_id: str
    title: str
    evidence_type: str
    matched_concepts: list[str] = Field(default_factory=list)
    support_snippet: str


class RequirementEvidenceMap(BaseModel):
    """Mapping between one extracted requirement and candidate evidence."""

    requirement_id: str
    requirement_text: str
    priority: RequirementPriority
    category: RequirementCategory
    support_level: SupportLevel
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    evidence: list[EvidenceMatch] = Field(default_factory=list)
    rationale: str


class EvidenceMappingResult(BaseModel):
    """Structured output for evidence mapping."""

    mappings: list[RequirementEvidenceMap] = Field(default_factory=list)


class RequirementCoverageScore(BaseModel):
    """Rubric score for one mapped requirement."""

    requirement_id: str
    requirement_text: str
    priority: RequirementPriority
    category: RequirementCategory
    support_level: SupportLevel
    coverage_status: CoverageStatus
    weight: float
    score: float
    max_score: float
    evidence_ids: list[str] = Field(default_factory=list)
    gap_summary: str


class CoverageScoringResult(BaseModel):
    """Structured output for rubric-based coverage scoring."""

    requirement_scores: list[RequirementCoverageScore] = Field(default_factory=list)
    total_score: float
    max_score: float
    coverage_percent: float
    supported_count: int
    partial_count: int
    unsupported_count: int
    policy_count: int
    gap_requirement_ids: list[str] = Field(default_factory=list)
    partial_requirement_ids: list[str] = Field(default_factory=list)
    summary: str


class ClaimEvidenceSupport(BaseModel):
    """Evidence item supporting a reviewed claim."""

    evidence_id: str
    title: str
    evidence_type: str
    matched_concepts: list[str] = Field(default_factory=list)
    support_snippet: str


class ClaimGroundednessCheck(BaseModel):
    """Groundedness result for one candidate-facing claim."""

    claim_id: str
    claim: str
    status: ClaimStatus
    severity: ClaimSeverity
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    evidence: list[ClaimEvidenceSupport] = Field(default_factory=list)
    explanation: str
    safe_action: str


class GroundednessCheckResult(BaseModel):
    """Structured output for unsupported-claim detection."""

    checks: list[ClaimGroundednessCheck] = Field(default_factory=list)
    unsupported_claims: list[ClaimGroundednessCheck] = Field(default_factory=list)
    grounded_count: int
    partial_count: int
    unsupported_count: int
    contradicted_count: int
    summary: str


class ApprovalGateResult(BaseModel):
    """Human approval decision for safe-only final drafting."""

    approval_status: ApprovalDecisionStatus
    can_draft: bool
    human_approved: bool
    grounded_claim_ids: list[str] = Field(default_factory=list)
    approved_claim_ids: list[str] = Field(default_factory=list)
    blocked_claim_ids: list[str] = Field(default_factory=list)
    draft_constraints: list[str] = Field(default_factory=list)
    required_human_action: str
    summary: str


class FinalDraftResult(BaseModel):
    """Safe final draft generated after human approval."""

    draft_text: str
    included_claim_ids: list[str] = Field(default_factory=list)
    excluded_claim_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
