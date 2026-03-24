#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_c2.sh \
    --model-name MODEL \
    --dataset-path DATASET \
    --output-root OUTPUT_DIR \
    [--eval-dataset-path DATASET] \
    [--max-retry-for-synthesis N] \
    [--deepspeed CONFIG_OR_AUTO] \
    [--zero-stage N] \
    [--cache-dir DIR] \
    [--tensor-parallel-size N] \
    [--gpu-memory-utilization FLOAT] \
    [--train-gpu-memory-utilization FLOAT] \
    [--max-model-len N]

This script runs the full C2 pipeline:
1. Synthesize helpful/misleading rubrics.
2. Train the rubric generator with DPO.
3. Train the verifier with mixed rubric-free and rubric-augmented GRPO.
4. Optionally run selective inference on an evaluation dataset.
EOF
}

MODEL_NAME=""
DATASET_PATH=""
OUTPUT_ROOT=""
EVAL_DATASET_PATH=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.uv-cache}"
CACHE_DIR="/data_pwftms01/kawabata/cache"
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
TRAIN_GPU_MEMORY_UTILIZATION=0.5
MAX_RETRY_FOR_SYNTHESIS=5
DEEPSPEED_CONFIG="auto"
ZERO_STAGE=2
MAX_MODEL_LEN=16384
DATASET_SPLIT="train"
EVAL_DATASET_SPLIT="test"
SEED=13

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --eval-dataset-path)
      EVAL_DATASET_PATH="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --max-retry-for-synthesis)
      MAX_RETRY_FOR_SYNTHESIS="$2"
      shift 2
      ;;
    --deepspeed)
      DEEPSPEED_CONFIG="$2"
      shift 2
      ;;
    --zero-stage)
      ZERO_STAGE="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --train-gpu-memory-utilization)
      TRAIN_GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --dataset-split)
      DATASET_SPLIT="$2"
      shift 2
      ;;
    --eval-dataset-split)
      EVAL_DATASET_SPLIT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_NAME" || -z "$DATASET_PATH" || -z "$OUTPUT_ROOT" ]]; then
  usage >&2
  exit 1
fi

DATA_DIR="${OUTPUT_ROOT%/}/data"
GENERATOR_DIR="${OUTPUT_ROOT%/}/generator"
VERIFIER_DIR="${OUTPUT_ROOT%/}/verifier"
CONTRASTIVE_RUBRIC_PAIRS_PATH="${DATA_DIR}/contrastive_rubric_pairs.jsonl"
GENERATOR_CONTRASTIVE_PAIRS_PATH="${DATA_DIR}/generator_contrastive_pairs.jsonl"
RUBRIC_AUGMENTED_EXAMPLES_PATH="${DATA_DIR}/rubric_augmented_examples.jsonl"
PREDICTIONS_PATH="${OUTPUT_ROOT%/}/predictions.jsonl"

mkdir -p "$DATA_DIR" "$GENERATOR_DIR" "$VERIFIER_DIR"
mkdir -p "$UV_CACHE_DIR"

uv run --project "$PROJECT_ROOT" c2-synthesize-rubrics \
  --dataset-path "$DATASET_PATH" \
  --dataset-split "$DATASET_SPLIT" \
  --model-name "$MODEL_NAME" \
  --output-path "$CONTRASTIVE_RUBRIC_PAIRS_PATH" \
  --generator-contrastive-pairs-path "$GENERATOR_CONTRASTIVE_PAIRS_PATH" \
  --rubric-augmented-examples-path "$RUBRIC_AUGMENTED_EXAMPLES_PATH" \
  --cache-dir "$CACHE_DIR" \
  --seed "$SEED" \
  --max-retry-for-synthesis "$MAX_RETRY_FOR_SYNTHESIS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"

uv run --project "$PROJECT_ROOT" c2-train-generator-dpo \
  --dataset-path "$GENERATOR_CONTRASTIVE_PAIRS_PATH" \
  --model-name "$MODEL_NAME" \
  --output-dir "$GENERATOR_DIR" \
  --cache-dir "$CACHE_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --zero-stage "$ZERO_STAGE" \
  --gradient-checkpointing \
  --use-flash-attention

uv run --project "$PROJECT_ROOT" c2-train-verifier-grpo \
  --original-dataset-path "$DATASET_PATH" \
  --dataset-split "$DATASET_SPLIT" \
  --rubric-augmented-examples-path "$RUBRIC_AUGMENTED_EXAMPLES_PATH" \
  --model-name "$MODEL_NAME" \
  --output-dir "$VERIFIER_DIR" \
  --cache-dir "$CACHE_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --zero-stage "$ZERO_STAGE" \
  --gradient-checkpointing \
  --bf16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$TRAIN_GPU_MEMORY_UTILIZATION"

if [[ -n "$EVAL_DATASET_PATH" ]]; then
  uv run --project "$PROJECT_ROOT" c2-infer \
    --dataset-path "$EVAL_DATASET_PATH" \
    --dataset-split "$EVAL_DATASET_SPLIT" \
    --generator-model "$GENERATOR_DIR" \
    --verifier-model "$VERIFIER_DIR" \
    --output-path "$PREDICTIONS_PATH" \
    --cache-dir "$CACHE_DIR" \
    --seed "$SEED" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN"
fi

echo "C2 pipeline finished. Outputs are under $OUTPUT_ROOT"
