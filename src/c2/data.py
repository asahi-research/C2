from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import load_dataset


def _validate_letter(value: str, field_name: str) -> str:
    """Normalize A/B labels and reject unsupported values."""

    normalized = value.strip().upper()
    if normalized not in {"A", "B"}:
        raise ValueError(f"{field_name} must be 'A' or 'B', got {value!r}")
    return normalized


def _validate_rubric_label(value: str) -> str:
    """Normalize helpful/misleading labels and reject unsupported values."""

    normalized = value.strip().lower()
    if normalized not in {"helpful", "misleading"}:
        raise ValueError(
            "rubric_label must be 'helpful' or 'misleading', "
            f"got {value!r}"
        )
    return normalized


@dataclass(frozen=True)
class PairwiseExample:
    """One pairwise preference example in canonical A/B form."""

    prompt_id: str
    prompt: str
    response_a: str
    response_b: str
    preferred_letter: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preferred_letter",
            _validate_letter(self.preferred_letter, "preferred_letter"),
        )

    @property
    def rejected_letter(self) -> str:
        """Return the non-preferred response label."""

        return "B" if self.preferred_letter == "A" else "A"


@dataclass(frozen=True)
class RubricCandidateScore:
    """One sampled rubric candidate paired with its verifier margin."""

    rubric_text: str
    margin: float


@dataclass(frozen=True)
class ContrastiveRubricPair:
    """Helpful/misleading rubric pair synthesized from one preference example."""

    prompt_id: str
    prompt: str
    response_a: str
    response_b: str
    preferred_letter: str
    base_margin: float
    helpful_rubric: str
    helpful_margin: float
    misleading_rubric: str
    misleading_margin: float
    candidate_margins: list[RubricCandidateScore] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preferred_letter",
            _validate_letter(self.preferred_letter, "preferred_letter"),
        )

    @property
    def rejected_letter(self) -> str:
        """Return the non-preferred response label."""

        return "B" if self.preferred_letter == "A" else "A"


@dataclass(frozen=True)
class RubricAugmentedExample:
    """One rubric-augmented verifier training example."""

    prompt_id: str
    prompt: str
    response_a: str
    response_b: str
    preferred_letter: str
    rubric_label: str
    rubric_text: str
    base_margin: float
    rubric_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preferred_letter",
            _validate_letter(self.preferred_letter, "preferred_letter"),
        )
        object.__setattr__(
            self,
            "rubric_label",
            _validate_rubric_label(self.rubric_label),
        )


def _require_json_object(value: Any, *, location: str) -> dict[str, Any]:
    """Validate one JSON row and normalize dataclasses into plain dicts."""

    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object at {location}, got {type(value).__name__}")
    return dict(value)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    resolved = Path(path)
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {resolved}") from exc
            records.append(
                _require_json_object(parsed, location=f"line {line_number} in {resolved}")
            )
    return records


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load either a JSON array file or a JSONL file."""

    resolved = Path(path)
    if resolved.suffix.lower() == ".jsonl":
        return load_jsonl(resolved)

    parsed = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array in {resolved}")
    return [
        _require_json_object(item, location=f"index {index} in {resolved}")
        for index, item in enumerate(parsed)
    ]


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Write records to a JSONL file."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _require_json_object(record, location="write_jsonl record"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def _require_non_empty_string(
    record: Mapping[str, Any],
    key: str,
    *,
    record_index: int,
) -> str:
    """Read one required string field from a pairwise example."""

    value = str(record.get(key) or "").strip()
    if value:
        return value
    raise ValueError(
        f"Record {record_index} is missing required field {key!r}. "
        "Expected prompt/response_a/response_b/label format."
    )


def canonicalize_pairwise_record(
    record: Mapping[str, Any],
    record_index: int,
    seed: int,
    *,
    randomize_positions: bool = True,
) -> PairwiseExample:
    """Convert one A/B pairwise record into the canonical schema used by C2."""

    prompt = _require_non_empty_string(record, "prompt", record_index=record_index)
    response_a = _require_non_empty_string(record, "response_a", record_index=record_index)
    response_b = _require_non_empty_string(record, "response_b", record_index=record_index)
    label = _validate_letter(
        _require_non_empty_string(record, "label", record_index=record_index),
        "label",
    )
    prompt_id = str(record.get("prompt_id") or record.get("id") or f"record_{record_index:06d}")

    rng = random.Random(seed + record_index)
    should_swap = randomize_positions and rng.random() < 0.5

    if should_swap:
        return PairwiseExample(
            prompt_id=prompt_id,
            prompt=prompt,
            response_a=response_b,
            response_b=response_a,
            preferred_letter="B" if label == "A" else "A",
        )
    return PairwiseExample(
        prompt_id=prompt_id,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        preferred_letter=label,
    )


def load_pairwise_dataset(
    dataset_path_or_name: str,
    *,
    split: str = "train",
    seed: int = 13,
    limit: int | None = None,
    randomize_positions: bool = True,
) -> list[PairwiseExample]:
    """Load and canonicalize a pairwise preference dataset."""

    dataset_path = Path(dataset_path_or_name)
    if dataset_path.exists():
        raw_records = load_json_or_jsonl(dataset_path)
    else:
        dataset = load_dataset(dataset_path_or_name, split=split)
        raw_records = [dict(item) for item in dataset]

    examples: list[PairwiseExample] = []
    for index, record in enumerate(raw_records):
        examples.append(
            canonicalize_pairwise_record(
                record,
                record_index=index,
                seed=seed,
                randomize_positions=randomize_positions,
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise ValueError("No usable pairwise examples were loaded")
    return examples


__all__ = [
    "ContrastiveRubricPair",
    "PairwiseExample",
    "RubricCandidateScore",
    "RubricAugmentedExample",
    "canonicalize_pairwise_record",
    "load_json_or_jsonl",
    "load_jsonl",
    "load_pairwise_dataset",
    "write_jsonl",
]
