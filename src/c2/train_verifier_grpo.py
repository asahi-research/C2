from __future__ import annotations

import argparse
import re
from typing import Any, Iterable

from datasets import Dataset, concatenate_datasets
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

from .data import PairwiseExample, load_jsonl, load_pairwise_dataset
from .deepspeed import build_deepspeed_config
from .llm import ensure_cache_dirs, load_tokenizer, render_chat_prompt
from .parsing import extract_answer_label, extract_rubric_label
from .prompts import (
    build_rubric_augmented_verification_prompt,
    build_rubric_free_verification_prompt,
)


RUBRIC_FREE_FORMAT_PATTERN = re.compile(
    r"^\s*<analyze>(?P<body>.+?)</analyze>\s*<answer>\s*([AB])\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
RUBRIC_AUGMENTED_FORMAT_PATTERN = re.compile(
    r"^\s*<analyze>(?P<body>.+?)</analyze>\s*<rubric>\s*(helpful|misleading)\s*</rubric>\s*<answer>\s*([AB])\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
EMBEDDED_TAG_PATTERN = re.compile(r"</?(analyze|answer|rubric)>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dataset-path", required=True, help="Pairwise dataset used for rubric-free tasks.")
    parser.add_argument(
        "--rubric-augmented-examples-path",
        required=True,
        help="Rubric-augmented examples emitted by C2 synthesis.",
    )
    parser.add_argument("--model-name", required=True, help="Base verifier model.")
    parser.add_argument("--output-dir", required=True, help="Directory where checkpoints are written.")
    parser.add_argument("--cache-dir", default="/data_pwftms01/kawabata/cache", help="Hugging Face cache directory.")
    parser.add_argument("--dataset-split", default="train", help="HF split for --original-dataset-path when needed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--shuffle-seed", type=int, default=1234, help="Seed used when shuffling training rows.")
    parser.add_argument("--max-training-records", type=int, default=None, help="Optional cap on total training rows.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-7, help="AdamW learning rate.")
    parser.add_argument("--lr-scheduler-type", default="linear", help="Learning rate schedule.")
    parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of GRPO epochs.")
    parser.add_argument("--beta", type=float, default=0.01, help="GRPO KL coefficient.")
    parser.add_argument("--num-generations", type=int, default=8, help="Number of sampled completions per prompt.")
    parser.add_argument("--generation-batch-size", type=int, default=64, help="vLLM generation batch size.")
    parser.add_argument("--max-prompt-length", type=int, default=8192, help="Maximum prompt length.")
    parser.add_argument("--max-completion-length", type=int, default=2048, help="Maximum completion length.")
    parser.add_argument("--generation-temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--generation-top-p", type=float, default=1.0, help="Sampling top-p.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="Fraction of GPU memory vLLM may reserve.",
    )
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging frequency.")
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="steps", help="Save cadence.")
    parser.add_argument("--save-steps", type=int, default=50, help="Save frequency for save-strategy=steps.")
    parser.add_argument("--save-total-limit", type=int, default=10, help="Maximum checkpoints to keep.")
    parser.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="no", help="Eval cadence.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Eval frequency for eval-strategy=steps.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument(
        "--deepspeed",
        default="auto",
        help="DeepSpeed JSON path or 'auto'. Pass an empty string to disable DeepSpeed.",
    )
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), default=2, help="ZeRO stage for --deepspeed auto.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable remote model code.")
    parser.add_argument("--attn-implementation", default="flash_attention_2", help="Attention backend.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--report-to", nargs="+", default=["tensorboard"], help="Reporting integrations.")
    parser.add_argument("--run-name", default=None, help="Optional experiment name.")
    parser.add_argument("--log-completions", action="store_true", help="Log sample completions.")
    parser.add_argument("--num-completions-to-print", type=int, default=8, help="How many completions to print.")
    parser.add_argument(
        "--reward-weights",
        type=float,
        nargs=3,
        default=[0.6, 0.3, 0.1],
        help="Weights for [preference, rubric, format] rewards.",
    )
    return parser.parse_args()


def _completion_to_text(completion: object) -> str:
    """Normalize GRPO completions to raw text."""

    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if completion is None:
        return ""
    return str(completion)


def _make_verifier_row(
    *,
    tokenizer: Any,
    model_name: str,
    prompt_id: str,
    user_prompt: str,
    target_letter: str,
    rubric_label: str,
    task_type: str,
) -> dict[str, str]:
    """Build one verifier training row in the shared GRPO schema."""

    return {
        "prompt": render_chat_prompt(tokenizer, user_prompt, model_name=model_name),
        "prompt_id": prompt_id,
        "target_letter": target_letter,
        "rubric_label": rubric_label,
        "task_type": task_type,
    }


def _build_original_dataset(
    *,
    dataset_path: str,
    split: str,
    tokenizer: Any,
    model_name: str,
    seed: int,
) -> Dataset:
    """Build rubric-free verifier rows from the original pairwise dataset."""

    rows: list[dict[str, Any]] = []
    examples = load_pairwise_dataset(
        dataset_path,
        split=split,
        seed=seed,
        randomize_positions=True,
    )
    for example in examples:
        rows.append(
            _make_verifier_row(
                tokenizer=tokenizer,
                model_name=model_name,
                prompt_id=f"{example.prompt_id}_original",
                user_prompt=build_rubric_free_verification_prompt(example),
                target_letter=example.preferred_letter,
                rubric_label="",
                task_type="rubric_free",
            )
        )
    return Dataset.from_list(rows)


def _build_rubric_augmented_dataset(
    *,
    rubric_augmented_examples_path: str,
    tokenizer: Any,
    model_name: str,
) -> Dataset:
    """Build rubric-augmented verifier rows from synthesized rubric-augmented examples."""

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(load_jsonl(rubric_augmented_examples_path)):
        rubric_text = str(record.get("rubric_text") or record.get("rubric") or "").strip()
        if not rubric_text:
            raise ValueError(f"Rubric-augmented example {index} is missing rubric_text")
        example = PairwiseExample(
            prompt_id=str(record.get("prompt_id") or f"rubric_augmented_example_{index:06d}"),
            prompt=str(record.get("prompt") or "").strip(),
            response_a=str(record.get("response_a") or "").strip(),
            response_b=str(record.get("response_b") or "").strip(),
            preferred_letter=str(record.get("preferred_letter") or record.get("target_letter") or "").strip(),
        )
        rows.append(
            _make_verifier_row(
                tokenizer=tokenizer,
                model_name=model_name,
                prompt_id=example.prompt_id,
                user_prompt=build_rubric_augmented_verification_prompt(example, rubric_text),
                target_letter=example.preferred_letter,
                rubric_label=str(record.get("rubric_label") or "").strip().lower(),
                task_type="rubric_augmented",
            )
        )
    return Dataset.from_list(rows)


def preference_reward(
    *,
    prompts: Iterable[str],
    completions: Iterable[object],
    target_letter: list[str],
    **_: dict[str, Any],
) -> list[float]:
    """Reward correct pairwise answers with +1 and incorrect answers with -1."""

    _ = prompts
    rewards: list[float] = []
    for completion, expected in zip(completions, target_letter):
        predicted = extract_answer_label(_completion_to_text(completion))
        rewards.append(1.0 if predicted == expected else -1.0)
    return rewards


def rubric_reward(
    *,
    prompts: Iterable[str],
    completions: Iterable[object],
    rubric_label: list[str],
    task_type: list[str],
    **_: dict[str, Any],
) -> list[float]:
    """Reward correct helpful/misleading decisions on rubric tasks only."""

    _ = prompts
    rewards: list[float] = []
    for completion, expected, task in zip(completions, rubric_label, task_type):
        if task != "rubric_augmented":
            rewards.append(0.0)
            continue
        predicted = extract_rubric_label(_completion_to_text(completion))
        rewards.append(1.0 if predicted == expected else -1.0)
    return rewards


def format_reward(
    *,
    prompts: Iterable[str],
    completions: Iterable[object],
    task_type: list[str],
    **_: dict[str, Any],
) -> list[float]:
    """Reward outputs that follow the required XML layout."""

    _ = prompts
    rewards: list[float] = []
    for completion, task in zip(completions, task_type):
        text = _completion_to_text(completion)
        pattern = (
            RUBRIC_AUGMENTED_FORMAT_PATTERN
            if task == "rubric_augmented"
            else RUBRIC_FREE_FORMAT_PATTERN
        )
        match = pattern.match(text)
        body = (match.group("body") if match else "").strip()
        is_valid = bool(match) and bool(body) and not EMBEDDED_TAG_PATTERN.search(body)
        rewards.append(1.0 if is_valid else -1.0)
    return rewards


def main() -> None:
    """Train the verifier with mixed rubric-free and rubric-augmented GRPO tasks."""

    args = parse_args()
    cache_dir = ensure_cache_dirs(args.cache_dir)
    set_seed(args.seed)

    tokenizer = load_tokenizer(
        args.model_name,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    original_dataset = _build_original_dataset(
        dataset_path=args.original_dataset_path,
        split=args.dataset_split,
        tokenizer=tokenizer,
        model_name=args.model_name,
        seed=args.shuffle_seed,
    )
    rubric_augmented_dataset = _build_rubric_augmented_dataset(
        rubric_augmented_examples_path=args.rubric_augmented_examples_path,
        tokenizer=tokenizer,
        model_name=args.model_name,
    )
    combined = concatenate_datasets([original_dataset, rubric_augmented_dataset]).shuffle(
        seed=args.shuffle_seed
    )

    if args.max_training_records is not None:
        combined = combined.select(range(min(args.max_training_records, len(combined))))

    rubric_free_count = sum(1 for row in combined if row["task_type"] == "rubric_free")
    rubric_augmented_count = sum(1 for row in combined if row["task_type"] == "rubric_augmented")
    print(f"Original rubric-free rows: {rubric_free_count:,}")
    print(f"Rubric-augmented rows:     {rubric_augmented_count:,}")
    print(f"Total GRPO rows:           {len(combined):,}")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        deepspeed=build_deepspeed_config(
            deepspeed=args.deepspeed,
            zero_stage=args.zero_stage,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            bf16=args.bf16,
            fp16=args.fp16,
        ),
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        beta=args.beta,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=args.tensor_parallel_size,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.generation_temperature,
        top_p=args.generation_top_p,
        bf16=args.bf16,
        fp16=args.fp16,
        model_init_kwargs={
            "cache_dir": cache_dir,
            "trust_remote_code": args.trust_remote_code,
            "attn_implementation": args.attn_implementation,
            "torch_dtype": "bfloat16" if args.bf16 else ("float16" if args.fp16 else "auto"),
        },
        reward_weights=args.reward_weights,
        report_to=args.report_to,
        run_name=args.run_name,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        seed=args.seed,
        data_seed=args.shuffle_seed,
        shuffle_dataset=True,
        vllm_enable_sleep_mode=True,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[preference_reward, rubric_reward, format_reward],
        args=training_args,
        train_dataset=combined,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    print(f"Saved verifier checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
