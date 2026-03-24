from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer


def ensure_cache_dirs(base_cache: str | Path) -> str:
    """Prepare Hugging Face cache directories before model loading."""

    cache_dir = str(base_cache)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
    return cache_dir


def load_tokenizer(
    model_name: str,
    *,
    cache_dir: str,
    trust_remote_code: bool = False,
) -> AutoTokenizer:
    """Load a tokenizer and ensure it has a pad token."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_chat_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    *,
    model_name: str | None,
) -> str:
    """Render one user prompt through the model's chat template."""

    chat_kwargs = {"enable_thinking": True} if model_name and "qwen3" in model_name.lower() else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )


def generate_in_batches(
    *,
    llm: Any,
    prompts: Sequence[str],
    sampling_params: Any,
    batch_size: int,
) -> list[Any]:
    """Run vLLM generation in prompt batches to keep memory use predictable."""

    outputs: list[Any] = []
    for start in range(0, len(prompts), batch_size):
        outputs.extend(llm.generate(list(prompts[start : start + batch_size]), sampling_params))
    return outputs


def build_llm(
    model_name: str,
    *,
    cache_dir: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    trust_remote_code: bool = False,
) -> Any:
    """Instantiate a vLLM engine with the shared C2 defaults."""

    from vllm import LLM

    return LLM(
        model=model_name,
        download_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )
