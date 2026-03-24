from __future__ import annotations

from typing import Any

from .data import RubricCandidateScore


def extract_letter_logprobs(
    candidate_output: Any,
    answer_label: str | None,
) -> dict[str, float]:
    """Extract A/B log-probabilities from one vLLM output.

    The generation prompt ends immediately before the judge answer, so the A/B
    distribution usually appears on the token right after ``answer``.
    """

    letter_logprobs: dict[str, float] = {}
    logprob_steps = getattr(candidate_output, "logprobs", None) or []
    token_ids = getattr(candidate_output, "token_ids", None) or []

    def _extract_letter(decoded: str | None) -> str | None:
        if not decoded:
            return None
        normalized = (
            decoded.strip()
            .replace("<", "")
            .replace(">", "")
            .replace("/", "")
            .strip()
        )
        if normalized in {"A", "B"}:
            return normalized
        return None

    def _collect(step: dict[int, Any]) -> None:
        for info in step.values():
            decoded = getattr(info, "decoded_token", None)
            letter = _extract_letter(decoded)
            if letter and letter not in letter_logprobs:
                letter_logprobs[letter] = float(getattr(info, "logprob", float("-inf")))

    for index, token_id in enumerate(token_ids):
        if index >= len(logprob_steps):
            break
        step = logprob_steps[index]
        if not isinstance(step, dict):
            continue

        entry = step.get(token_id)
        if entry is None:
            continue
        decoded = getattr(entry, "decoded_token", None)

        if decoded and decoded.strip() == "answer":
            next_index = index + 1
            if next_index < len(logprob_steps):
                next_step = logprob_steps[next_index]
                if isinstance(next_step, dict):
                    _collect(next_step)
                    if len(letter_logprobs) >= 2:
                        break

        letter = _extract_letter(decoded)
        if letter and letter not in letter_logprobs:
            letter_logprobs[letter] = float(getattr(entry, "logprob", float("-inf")))
            if len(letter_logprobs) >= 2:
                break

    if len(letter_logprobs) < 2:
        for step in logprob_steps:
            if isinstance(step, dict):
                _collect(step)
            if len(letter_logprobs) >= 2:
                break

    if answer_label and answer_label not in letter_logprobs:
        for index, token_id in enumerate(token_ids):
            if index >= len(logprob_steps):
                break
            step = logprob_steps[index]
            if not isinstance(step, dict):
                continue
            entry = step.get(token_id)
            if entry is None:
                continue
            decoded = getattr(entry, "decoded_token", None)
            letter = _extract_letter(decoded)
            if letter == answer_label:
                letter_logprobs[letter] = float(getattr(entry, "logprob", float("-inf")))
                break

    return letter_logprobs


def compute_margin(
    *,
    preferred_letter: str,
    letter_logprobs: dict[str, float],
) -> float | None:
    """Compute the judge margin for the preferred label."""

    rejected_letter = "B" if preferred_letter == "A" else "A"
    preferred_logprob = letter_logprobs.get(preferred_letter, float("-inf"))
    rejected_logprob = letter_logprobs.get(rejected_letter, float("-inf"))
    if preferred_logprob == float("-inf") and rejected_logprob == float("-inf"):
        return None
    return preferred_logprob - rejected_logprob


def select_contrastive_rubrics(
    *,
    base_margin: float,
    candidates: list[RubricCandidateScore],
) -> tuple[RubricCandidateScore, RubricCandidateScore] | None:
    """Select the paper's helpful/misleading rubric pair from candidate margins."""

    helpful_threshold = max(0.0, base_margin)
    misleading_threshold = min(0.0, base_margin)

    helpful = [candidate for candidate in candidates if candidate.margin > helpful_threshold]
    misleading = [
        candidate
        for candidate in candidates
        if candidate.margin < misleading_threshold
    ]

    if not helpful or not misleading:
        return None

    helpful_best = max(helpful, key=lambda candidate: candidate.margin)
    misleading_best = min(misleading, key=lambda candidate: candidate.margin)
    return helpful_best, misleading_best
