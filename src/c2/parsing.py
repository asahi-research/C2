from __future__ import annotations

import re


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
RUBRIC_TAG_PATTERN = re.compile(
    r"<rubric>\s*(helpful|misleading)\s*</rubric>",
    re.IGNORECASE,
)


def extract_answer_label(text: str) -> str | None:
    """Extract the final A/B answer label from model output."""

    if not text:
        return None
    matches = ANSWER_TAG_PATTERN.findall(text)
    if matches:
        return matches[-1].upper()
    stripped = text.strip().upper()
    if stripped in {"A", "B"}:
        return stripped
    return None


def extract_rubric_label(text: str) -> str | None:
    """Extract the helpful/misleading rubric label from model output."""

    if not text:
        return None
    matches = RUBRIC_TAG_PATTERN.findall(text)
    if matches:
        return matches[-1].lower()
    return None
