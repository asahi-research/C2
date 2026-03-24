from __future__ import annotations

import argparse
from typing import Any, Iterable, Sequence

from vllm import SamplingParams

from .data import (
    ContrastiveRubricPair,
    PairwiseExample,
    RubricAugmentedExample,
    RubricCandidateScore,
    load_pairwise_dataset,
    write_jsonl,
)
from .llm import (
    build_llm,
    ensure_cache_dirs,
    generate_in_batches,
    load_tokenizer,
    render_chat_prompt,
)
from .parsing import extract_answer_label
from .prompts import (
    build_rubric_conditioned_selection_prompt,
    build_rubric_free_verification_prompt,
    build_rubric_generation_prompt,
)
from .scoring import compute_margin, extract_letter_logprobs, select_contrastive_rubrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="Local JSON/JSONL path or HF dataset name.")
    parser.add_argument("--dataset-split", default="train", help="HF dataset split when --dataset-path is a dataset name.")
    parser.add_argument("--model-name", required=True, help="Base model used as both M_g and M_v.")
    parser.add_argument("--output-path", required=True, help="Where to write contrastive rubric pairs.")
    parser.add_argument(
        "--generator-contrastive-pairs-path",
        default="",
        help="Optional generator contrastive pairs path written alongside synthesis.",
    )
    parser.add_argument(
        "--rubric-augmented-examples-path",
        default="",
        help="Optional rubric-augmented example path written alongside synthesis.",
    )
    parser.add_argument("--cache-dir", default="/data_pwftms01/kawabata/cache", help="Hugging Face cache directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on loaded records.")
    parser.add_argument("--seed", type=int, default=13, help="Seed used when randomizing A/B order.")
    parser.add_argument("--num-rubrics", type=int, default=16, help="Number of rubric candidates sampled per example.")
    parser.add_argument(
        "--max-retry-for-synthesis",
        type=int,
        default=5,
        help="How many extra rubric-generation retries to run for unresolved examples.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Number of prompts per vLLM batch.")
    parser.add_argument("--rubric-temperature", type=float, default=1.0, help="Sampling temperature for rubric generation.")
    parser.add_argument("--rubric-top-p", type=float, default=0.9, help="Top-p for rubric generation.")
    parser.add_argument("--rubric-max-tokens", type=int, default=1024, help="Maximum rubric generation length.")
    parser.add_argument("--judge-max-tokens", type=int, default=512, help="Maximum verifier generation length.")
    parser.add_argument("--answer-logprobs", type=int, default=20, help="How many token logprobs to request from vLLM.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM may reserve.",
    )
    parser.add_argument("--max-model-len", type=int, default=16384, help="Maximum model context length.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable remote model code.")
    return parser.parse_args()


def _score_margin_from_output(output: Any, preferred_letter: str) -> float | None:
    """Extract the preferred-vs-rejected margin from one verifier output."""

    if not output.outputs:
        return None
    candidate_output = output.outputs[0]
    generated_text = candidate_output.text.strip()
    answer_label = extract_answer_label(generated_text)
    letter_logprobs = extract_letter_logprobs(candidate_output, answer_label)
    return compute_margin(
        preferred_letter=preferred_letter,
        letter_logprobs=letter_logprobs,
    )


def _build_generator_contrastive_pairs(
    records: Iterable[ContrastiveRubricPair],
) -> list[dict[str, Any]]:
    """Convert contrastive rubric pairs into DPO chat examples for the generator."""

    rows: list[dict[str, Any]] = []
    for record in records:
        example = PairwiseExample(
            prompt_id=record.prompt_id,
            prompt=record.prompt,
            response_a=record.response_a,
            response_b=record.response_b,
            preferred_letter=record.preferred_letter,
        )
        user_prompt = build_rubric_generation_prompt(example)
        rows.append(
            {
                "prompt_id": record.prompt_id,
                "chosen": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": record.helpful_rubric},
                ],
                "rejected": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": record.misleading_rubric},
                ],
                "base_margin": record.base_margin,
                "helpful_margin": record.helpful_margin,
                "misleading_margin": record.misleading_margin,
            }
        )
    return rows


def _build_rubric_augmented_examples(
    records: Iterable[ContrastiveRubricPair],
) -> list[RubricAugmentedExample]:
    """Convert synthesized pairs into rubric-augmented verifier examples."""

    rows: list[RubricAugmentedExample] = []
    for record in records:
        rows.append(
            RubricAugmentedExample(
                prompt_id=f"{record.prompt_id}_helpful",
                prompt=record.prompt,
                response_a=record.response_a,
                response_b=record.response_b,
                preferred_letter=record.preferred_letter,
                rubric_label="helpful",
                rubric_text=record.helpful_rubric,
                base_margin=record.base_margin,
                rubric_margin=record.helpful_margin,
            )
        )
        rows.append(
            RubricAugmentedExample(
                prompt_id=f"{record.prompt_id}_misleading",
                prompt=record.prompt,
                response_a=record.response_a,
                response_b=record.response_b,
                preferred_letter=record.preferred_letter,
                rubric_label="misleading",
                rubric_text=record.misleading_rubric,
                base_margin=record.base_margin,
                rubric_margin=record.misleading_margin,
            )
        )
    return rows


def _sample_and_score_rubrics(
    *,
    examples: Sequence[PairwiseExample],
    pending_indices: Sequence[int],
    tokenizer: Any,
    model_name: str,
    llm: Any,
    rubric_sampling: SamplingParams,
    judge_sampling: SamplingParams,
    batch_size: int,
    seen_rubrics_by_example: Sequence[set[str]],
) -> tuple[dict[int, list[RubricCandidateScore]], int]:
    """Sample rubric candidates for pending examples and score only new rubrics."""

    if not pending_indices:
        return {}, 0

    generation_prompts = [
        render_chat_prompt(
            tokenizer,
            build_rubric_generation_prompt(examples[example_index]),
            model_name=model_name,
        )
        for example_index in pending_indices
    ]
    generation_outputs = generate_in_batches(
        llm=llm,
        prompts=generation_prompts,
        sampling_params=rubric_sampling,
        batch_size=batch_size,
    )

    scoring_prompts: list[str] = []
    scoring_index: list[tuple[int, str]] = []
    raw_rubric_count = 0

    for example_index, output in zip(pending_indices, generation_outputs, strict=True):
        seen_rubrics = seen_rubrics_by_example[example_index]
        for candidate in output.outputs:
            rubric_text = candidate.text.strip()
            if not rubric_text:
                continue
            raw_rubric_count += 1
            if rubric_text in seen_rubrics:
                continue
            seen_rubrics.add(rubric_text)
            scoring_prompts.append(
                render_chat_prompt(
                    tokenizer,
                    build_rubric_conditioned_selection_prompt(examples[example_index], rubric_text),
                    model_name=model_name,
                )
            )
            scoring_index.append((example_index, rubric_text))

    if not scoring_prompts:
        return {}, raw_rubric_count

    scoring_outputs = generate_in_batches(
        llm=llm,
        prompts=scoring_prompts,
        sampling_params=judge_sampling,
        batch_size=batch_size,
    )

    scored_candidates: dict[int, list[RubricCandidateScore]] = {}
    for output, (example_index, rubric_text) in zip(scoring_outputs, scoring_index, strict=True):
        margin = _score_margin_from_output(output, examples[example_index].preferred_letter)
        if margin is None:
            continue
        scored_candidates.setdefault(example_index, []).append(
            RubricCandidateScore(rubric_text=rubric_text, margin=margin)
        )

    return scored_candidates, raw_rubric_count


def _resolve_pending_examples(
    *,
    pending_indices: Sequence[int],
    base_margins: Sequence[float | None],
    scored_candidates_by_example: Sequence[list[RubricCandidateScore]],
) -> tuple[dict[int, tuple[RubricCandidateScore, RubricCandidateScore]], list[int]]:
    """Resolve pending examples and return the unresolved subset for the next retry."""

    resolved: dict[int, tuple[RubricCandidateScore, RubricCandidateScore]] = {}
    still_pending: list[int] = []

    for example_index in pending_indices:
        base_margin = base_margins[example_index]
        if base_margin is None:
            continue
        selected = select_contrastive_rubrics(
            base_margin=base_margin,
            candidates=scored_candidates_by_example[example_index],
        )
        if selected is None:
            still_pending.append(example_index)
            continue
        resolved[example_index] = selected

    return resolved, still_pending


def main() -> None:
    """Run rubric synthesis, then optionally emit DPO and verifier datasets."""

    args = parse_args()
    cache_dir = ensure_cache_dirs(args.cache_dir)
    if args.max_retry_for_synthesis < 0:
        raise ValueError("--max-retry-for-synthesis must be >= 0")

    examples = load_pairwise_dataset(
        args.dataset_path,
        split=args.dataset_split,
        seed=args.seed,
        limit=args.limit,
        randomize_positions=True,
    )
    print(f"Loaded {len(examples):,} pairwise examples")

    tokenizer = load_tokenizer(
        args.model_name,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    llm = build_llm(
        args.model_name,
        cache_dir=cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    baseline_prompts = [
        render_chat_prompt(
            tokenizer,
            build_rubric_free_verification_prompt(example),
            model_name=args.model_name,
        )
        for example in examples
    ]
    judge_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.judge_max_tokens,
        logprobs=args.answer_logprobs,
    )
    baseline_outputs = generate_in_batches(
        llm=llm,
        prompts=baseline_prompts,
        sampling_params=judge_sampling,
        batch_size=args.batch_size,
    )
    base_margins = [
        _score_margin_from_output(output, example.preferred_letter)
        for output, example in zip(baseline_outputs, examples, strict=True)
    ]

    rubric_sampling = SamplingParams(
        n=args.num_rubrics,
        temperature=args.rubric_temperature,
        top_p=args.rubric_top_p,
        max_tokens=args.rubric_max_tokens,
    )
    scored_candidates_by_example: list[list[RubricCandidateScore]] = [[] for _ in examples]
    seen_rubrics_by_example: list[set[str]] = [set() for _ in examples]
    selected_pairs_by_example: dict[int, tuple[RubricCandidateScore, RubricCandidateScore]] = {}
    pending_indices = [
        example_index
        for example_index, base_margin in enumerate(base_margins)
        if base_margin is not None
    ]

    total_attempts = args.max_retry_for_synthesis + 1
    for attempt_index in range(total_attempts):
        if not pending_indices:
            break

        scored_candidates, raw_rubric_count = _sample_and_score_rubrics(
            examples=examples,
            pending_indices=pending_indices,
            tokenizer=tokenizer,
            model_name=args.model_name,
            llm=llm,
            rubric_sampling=rubric_sampling,
            judge_sampling=judge_sampling,
            batch_size=args.batch_size,
            seen_rubrics_by_example=seen_rubrics_by_example,
        )
        for example_index, new_candidates in scored_candidates.items():
            scored_candidates_by_example[example_index].extend(new_candidates)

        resolved_now, next_pending = _resolve_pending_examples(
            pending_indices=pending_indices,
            base_margins=base_margins,
            scored_candidates_by_example=scored_candidates_by_example,
        )
        selected_pairs_by_example.update(resolved_now)

        attempt_label = "initial pass" if attempt_index == 0 else f"retry {attempt_index}/{args.max_retry_for_synthesis}"
        new_candidate_count = sum(len(candidates) for candidates in scored_candidates.values())
        print(
            f"Synthesis {attempt_label}: "
            f"sampled {raw_rubric_count:,} raw rubrics, "
            f"scored {new_candidate_count:,} new rubrics, "
            f"resolved {len(resolved_now):,} examples, "
            f"{len(next_pending):,} still pending"
        )
        pending_indices = next_pending

    contrastive_rubric_pairs: list[ContrastiveRubricPair] = []
    skipped_no_base_margin = 0
    skipped_no_pair = 0

    for example_index, (example, base_margin, candidates) in enumerate(
        zip(
            examples,
            base_margins,
            scored_candidates_by_example,
            strict=True,
        )
    ):
        if base_margin is None:
            skipped_no_base_margin += 1
            continue
        selected = selected_pairs_by_example.get(example_index)
        if selected is None:
            skipped_no_pair += 1
            continue
        helpful, misleading = selected
        contrastive_rubric_pairs.append(
            ContrastiveRubricPair(
                prompt_id=example.prompt_id,
                prompt=example.prompt,
                response_a=example.response_a,
                response_b=example.response_b,
                preferred_letter=example.preferred_letter,
                base_margin=base_margin,
                helpful_rubric=helpful.rubric_text,
                helpful_margin=helpful.margin,
                misleading_rubric=misleading.rubric_text,
                misleading_margin=misleading.margin,
                candidate_margins=list(candidates),
            )
        )

    write_jsonl(args.output_path, contrastive_rubric_pairs)

    if args.generator_contrastive_pairs_path:
        generator_contrastive_pairs = _build_generator_contrastive_pairs(contrastive_rubric_pairs)
        write_jsonl(args.generator_contrastive_pairs_path, generator_contrastive_pairs)
        print(
            "Wrote "
            f"{len(generator_contrastive_pairs):,} generator contrastive pairs to "
            f"{args.generator_contrastive_pairs_path}"
        )

    if args.rubric_augmented_examples_path:
        rubric_augmented_examples = _build_rubric_augmented_examples(contrastive_rubric_pairs)
        write_jsonl(args.rubric_augmented_examples_path, rubric_augmented_examples)
        print(
            "Wrote "
            f"{len(rubric_augmented_examples):,} rubric-augmented examples to "
            f"{args.rubric_augmented_examples_path}"
        )

    print(f"Synthesized {len(contrastive_rubric_pairs):,} contrastive rubric pairs")
    print(f"Skipped {skipped_no_base_margin:,} examples because the base margin could not be scored")
    print(f"Skipped {skipped_no_pair:,} examples because helpful/misleading sets were empty")


if __name__ == "__main__":
    main()
