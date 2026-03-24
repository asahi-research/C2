from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def build_deepspeed_config(
    *,
    deepspeed: str,
    zero_stage: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    bf16: bool = False,
    fp16: bool = False,
) -> dict[str, Any] | None:
    """Return a DeepSpeed config dict.

    ``deepspeed`` may be:
    - ``""`` to disable DeepSpeed
    - ``"auto"`` to build a ZeRO config from the current CLI arguments
    - a path to a JSON config file
    """

    if not deepspeed:
        return None
    if deepspeed == "auto":
        world_size = max(int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1)), 1)
        config: dict[str, Any] = {
            "train_batch_size": per_device_train_batch_size * gradient_accumulation_steps * world_size,
            "train_micro_batch_size_per_gpu": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
        }
        if bf16:
            config["bf16"] = {"enabled": True}
        elif fp16:
            config["fp16"] = {"enabled": True}
        return config

    config_path = Path(deepspeed)
    if not config_path.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))
