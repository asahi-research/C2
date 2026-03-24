from __future__ import annotations

import argparse
from typing import Any

from vllm import SamplingParams

from .data import load_pairwise_dataset, write_jsonl
from .llm import (
    build_llm,
    ensure_cache_dirs,
    generate_in_batches,
    load_tokenizer,
    render_chat_prompt,
)
from .parsing import extract_answer_label, extract_rubric_label
from .prompts import (
    build_rubric_augmented_verification_prompt,
    build_rubric_free_verification_prompt,
    build_rubric_generation_prompt,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="Local JSON/JSONL path or HF dataset name.")
    parser.add_argument("--dataset-split", default="test", help="HF split when --dataset-path is a dataset name.")
    parser.add_argument("--generator-model", required=True, help="Trained rubric generator checkpoint.")
    parser.add_argument("--verifier-model", required=True, help="Trained verifier checkpoint.")
    parser.add_argument("--output-path", required=True, help="Where to write predictions.")
    parser.add_argument("--cache-dir", default="/data_pwftms01/kawabata/cache", help="Hugging Face cache directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated examples.")
    parser.add_argument("--seed", type=int, default=13, help="Seed used when randomizing A/B order.")
    parser.add_argument("--batch-size", type=int, default=64, help="vLLM prompt batch size.")
    parser.add_argument("--rubric-temperature", type=float, default=1.0, help="Sampling temperature for G_phi.")
    parser.add_argument("--rubric-top-p", type=float, default=0.9, help="Sampling top-p for G_phi.")
    parser.add_argument("--rubric-max-tokens", type=int, default=1024, help="Maximum rubric generation length.")
    parser.add_argument("--judge-max-tokens", type=int, default=1024, help="Maximum verifier generation length.")
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


def _first_text(output: Any) -> str:
    """Return the first generated text from one vLLM request."""

    if not output.outputs:
        return ""
    return output.outputs[0].text.strip()


def main() -> None:
    """Run C2 selective inference on a dataset."""

    args = parse_args()
    cache_dir = ensure_cache_dirs(args.cache_dir)

    examples = load_pairwise_dataset(
        args.dataset_path,
        split=args.dataset_split,
        seed=args.seed,
        limit=args.limit,
        randomize_positions=True,
    )
    print(f"Loaded {len(examples):,} evaluation examples")

    generator_tokenizer = load_tokenizer(
        args.generator_model,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    generator_llm = build_llm(
        args.generator_model,
        cache_dir=cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    if args.generator_model == args.verifier_model:
        verifier_tokenizer = generator_tokenizer
        verifier_llm = generator_llm
    else:
        verifier_tokenizer = load_tokenizer(
            args.verifier_model,
            cache_dir=cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        verifier_llm = build_llm(
            args.verifier_model,
            cache_dir=cache_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
        )

    generator_prompts = [
        render_chat_prompt(
            generator_tokenizer,
            build_rubric_generation_prompt(example),
            model_name=args.generator_model,
        )
        for example in examples
    ]
    rubric_sampling = SamplingParams(
        temperature=args.rubric_temperature,
        top_p=args.rubric_top_p,
        max_tokens=args.rubric_max_tokens,
    )
    rubric_outputs = generate_in_batches(
        llm=generator_llm,
        prompts=generator_prompts,
        sampling_params=rubric_sampling,
        batch_size=args.batch_size,
    )
    rubric_texts = [_first_text(output) for output in rubric_outputs]

    verifier_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.judge_max_tokens,
    )
    augmented_prompts = [
        render_chat_prompt(
            verifier_tokenizer,
            build_rubric_augmented_verification_prompt(example, rubric_text),
            model_name=args.verifier_model,
        )
        for example, rubric_text in zip(examples, rubric_texts, strict=True)
    ]
    augmented_outputs = generate_in_batches(
        llm=verifier_llm,
        prompts=augmented_prompts,
        sampling_params=verifier_sampling,
        batch_size=args.batch_size,
    )

    fallback_indices: list[int] = []
    augmented_answers: list[str | None] = []
    rubric_labels: list[str | None] = []
    augmented_texts: list[str] = []

    for index, output in enumerate(augmented_outputs):
        text = _first_text(output)
        augmented_texts.append(text)
        rubric_label = extract_rubric_label(text)
        answer = extract_answer_label(text)
        rubric_labels.append(rubric_label)
        augmented_answers.append(answer)
        if rubric_label != "helpful" or answer is None:
            fallback_indices.append(index)

    fallback_prompts = [
        render_chat_prompt(
            verifier_tokenizer,
            build_rubric_free_verification_prompt(examples[index]),
            model_name=args.verifier_model,
        )
        for index in fallback_indices
    ]
    fallback_outputs = generate_in_batches(
        llm=verifier_llm,
        prompts=fallback_prompts,
        sampling_params=verifier_sampling,
        batch_size=args.batch_size,
    )

    fallback_answers: dict[int, str | None] = {}
    fallback_texts: dict[int, str] = {}
    for index, output in zip(fallback_indices, fallback_outputs, strict=True):
        text = _first_text(output)
        fallback_texts[index] = text
        fallback_answers[index] = extract_answer_label(text)

    prediction_rows: list[dict[str, Any]] = []
    correct = 0
    fallback_count = 0

    for index, example in enumerate(examples):
        fallback_used = index in fallback_answers
        if fallback_used:
            fallback_count += 1
        final_answer = (
            augmented_answers[index]
            if rubric_labels[index] == "helpful" and augmented_answers[index] is not None
            else fallback_answers.get(index)
        )
        is_correct = final_answer == example.preferred_letter
        correct += int(is_correct)
        prediction_rows.append(
            {
                "prompt_id": example.prompt_id,
                "preferred_letter": example.preferred_letter,
                "rubric_text": rubric_texts[index],
                "rubric_label": rubric_labels[index],
                "augmented_answer": augmented_answers[index],
                "augmented_output": augmented_texts[index],
                "fallback_used": fallback_used,
                "fallback_answer": fallback_answers.get(index),
                "fallback_output": fallback_texts.get(index, ""),
                "final_answer": final_answer,
                "correct": is_correct,
            }
        )

    write_jsonl(args.output_path, prediction_rows)

    accuracy = correct / len(prediction_rows) if prediction_rows else 0.0
    fallback_rate = fallback_count / len(prediction_rows) if prediction_rows else 0.0
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fallback rate: {fallback_rate:.4f}")
    print(f"Wrote predictions to {args.output_path}")


if __name__ == "__main__":
    main()
