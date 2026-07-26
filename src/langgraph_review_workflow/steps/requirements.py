"""Requirement extraction step.

This module intentionally uses deterministic parsing for the portfolio demo. The
step is wrapped as a LangChain Core runnable so it can later be swapped for a
LiteLLM-compatible structured-output model boundary without changing the graph.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from langchain_core.runnables import RunnableLambda

from langgraph_review_workflow.schemas import (
    ExtractedRequirement,
    RequirementCategory,
    RequirementExtractionResult,
    RequirementPriority,
)


SECTION_ALIASES: dict[str, tuple[RequirementCategory, RequirementPriority] | None] = {
    "responsibilities": ("responsibility", "should_have"),
    "required qualifications": ("required_qualification", "must_have"),
    "nice to have": ("nice_to_have", "nice_to_have"),
    "language": ("language", "must_have"),
    "important note": ("claim_policy", "policy"),
}

STOPWORDS = {
    "a",
    "add",
    "against",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "can",
    "clear",
    "for",
    "from",
    "in",
    "is",
    "local",
    "of",
    "or",
    "should",
    "small",
    "that",
    "the",
    "to",
    "using",
    "we",
    "with",
}


def _normalize_heading(line: str) -> str:
    return line.strip().rstrip(":").lower()


def _split_sections(job_description: str) -> tuple[str, dict[str, list[str]]]:
    lines = [line.strip() for line in job_description.splitlines()]
    title = next((line for line in lines if line), "Untitled Role")
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in lines[1:]:
        if not line:
            continue

        normalized = _normalize_heading(line)
        if normalized in SECTION_ALIASES or normalized in {"company", "role summary"}:
            current_section = normalized
            sections.setdefault(current_section, [])
            continue

        if current_section:
            sections[current_section].append(line)

    return title, sections


def _clean_item(line: str) -> str:
    return line.removeprefix("-").strip()


def _iter_requirement_items(lines: Iterable[str]) -> Iterable[str]:
    paragraph_parts: list[str] = []

    for line in lines:
        if line.startswith("-"):
            if paragraph_parts:
                yield " ".join(paragraph_parts)
                paragraph_parts = []
            yield _clean_item(line)
        else:
            paragraph_parts.append(line)

    if paragraph_parts:
        yield " ".join(paragraph_parts)


def _extract_company(sections: dict[str, list[str]]) -> str | None:
    company_lines = sections.get("company", [])
    if not company_lines:
        return None
    first_line = company_lines[0]
    return first_line.split(" is ", maxsplit=1)[0].strip()


def _keywords_for(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+-]*", text.lower())
    keywords: list[str] = []

    for word in words:
        if len(word) < 3 or word in STOPWORDS or word in keywords:
            continue
        keywords.append(word)

    return keywords[:8]


def extract_requirements_from_text(job_description: str) -> RequirementExtractionResult:
    """Extract structured requirements from a sectioned job description."""

    role_title, sections = _split_sections(job_description)
    requirements: list[ExtractedRequirement] = []

    for section_name, lines in sections.items():
        classification = SECTION_ALIASES.get(section_name)
        if classification is None:
            continue

        category, priority = classification
        for item in _iter_requirement_items(lines):
            if not item:
                continue
            requirement_id = f"REQ-{len(requirements) + 1:03d}"
            requirements.append(
                ExtractedRequirement(
                    requirement_id=requirement_id,
                    category=category,
                    priority=priority,
                    text=item,
                    source_section=section_name.title(),
                    keywords=_keywords_for(item),
                )
            )

    return RequirementExtractionResult(
        role_title=role_title,
        company=_extract_company(sections),
        requirements=requirements,
    )


requirement_extraction_runnable = RunnableLambda(extract_requirements_from_text)
