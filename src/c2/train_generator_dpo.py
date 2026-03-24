# File: src/c2/train_generator_dpo.py
# Purpose: Fine-tune the C2 rubric generator with DPO.
"""Train the cooperative rubric generator on synthesized helpful/misleading pairs."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, set_seed
from transformers.training_args import IntervalStrategy, SaveStrategy
from trl import DPOConfig, DPOTrainer

from .deepspeed import build_deepspeed_config
from .llm import ensure_cache_dirs, load_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="JSONL file of generator contrastive pairs emitted by C2 synthesis.",
    )
    parser.add_argument("--model-name", required=True, help="Base model or SFT checkpoint to fine-tune.")
    parser.add_argument("--ref-model-name", default="", help="Optional reference model for DPO.")
    parser.add_argument("--output-dir", required=True, help="Directory where checkpoints are written.")
    parser.add_argument("--cache-dir", default="/data_pwftms01/kawabata/cache", help="Hugging Face cache directory.")
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of DPO epochs.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-7, help="AdamW learning rate.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Linear warmup ratio.")
    parser.add_argument("--max-length", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--eval-ratio", type=float, default=0.02, help="Held-out ratio. Set 0 to disable eval.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging frequency.")
    parser.add_argument(
        "--save-strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
        help="Checkpoint save schedule.",
    )
    parser.add_argument("--save-total-limit", type=int, default=3, help="How many checkpoints to keep.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--use-flash-attention", action="store_true", help="Enable FlashAttention when available.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable remote model code.")
    parser.add_argument(
        "--deepspeed",
        default="auto",
        help="DeepSpeed JSON path or 'auto'. Pass an empty string to disable DeepSpeed.",
    )
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), default=2, help="ZeRO stage for --deepspeed auto.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on DPO rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Validate DPO chat messages before feeding them into TRL."""

    if not isinstance(messages, list):
        raise ValueError("Expected a list of messages")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {index} is not an object")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if not role:
            raise ValueError(f"Message {index} is missing a role")
        normalized.append({"role": role, "content": content})
    return normalized


def main() -> None:
    """Train the rubric generator with DPO."""

    args = parse_args()
    cache_dir = ensure_cache_dirs(args.cache_dir)
    set_seed(args.seed)

    tokenizer = load_tokenizer(
        args.model_name,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    dataset = load_dataset(
        "json",
        data_files=args.dataset_path,
        split="train",
        cache_dir=str(Path(cache_dir) / "datasets"),
    )

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        """Validate chosen/rejected conversations."""

        return {
            "chosen": _normalize_messages(example["chosen"]),
            "rejected": _normalize_messages(example["rejected"]),
        }

    processed = dataset.map(
        preprocess,
        remove_columns=[column for column in dataset.column_names if column not in {"chosen", "rejected"}],
    ).shuffle(seed=args.seed)

    if args.limit is not None:
        processed = processed.select(range(min(args.limit, len(processed))))

    eval_ratio = max(0.0, min(args.eval_ratio, 0.5))
    if eval_ratio > 0:
        split = processed.train_test_split(test_size=eval_ratio, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = processed
        eval_dataset = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ref_model_name = args.ref_model_name or args.model_name
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )

    save_strategy = {
        "no": SaveStrategy.NO,
        "steps": SaveStrategy.STEPS,
        "epoch": SaveStrategy.EPOCH,
    }[args.save_strategy]

    training_args = DPOConfig(
        output_dir=args.output_dir,
        bf16=torch.cuda.is_available(),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        eval_strategy=IntervalStrategy.STEPS if eval_dataset is not None else IntervalStrategy.NO,
        do_eval=eval_dataset is not None,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=build_deepspeed_config(
            deepspeed=args.deepspeed,
            zero_stage=args.zero_stage,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            bf16=torch.cuda.is_available(),
        ),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved generator checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
