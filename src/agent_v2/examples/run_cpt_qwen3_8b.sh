#!/bin/bash
# CPT training script for Qwen3-8B on rocm-agent trajectory data
# Usage: bash run_cpt_qwen3_8b.sh

set -e

# Configuration
DATA_FILE="${DATA_FILE:-/home/jiajjiao/rocm-agent/training_data/large_dataset.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jiajjiao/rocm-agent/outputs/qwen3-8b-cpt}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
NUM_GPUS="${NUM_GPUS:-8}"

# Training hyperparameters
MAX_LENGTH="${MAX_LENGTH:-32768}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MIN_REWARD="${MIN_REWARD:-0.0}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPT_SCRIPT="${SCRIPT_DIR}/../cpt_qwen3_8b.py"
DS_CONFIG="${SCRIPT_DIR}/../ds_config_zero2.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "CPT Training for Qwen3-8B"
echo "=========================================="
echo "Data file: ${DATA_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model: ${MODEL}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Max length: ${MAX_LENGTH}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accum: ${GRAD_ACCUM}"
echo "Learning rate: ${LR}"
echo "Min reward: ${MIN_REWARD}"
echo "=========================================="

# Run training with DeepSpeed
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    "${CPT_SCRIPT}" \
    --data-file "${DATA_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --max-length ${MAX_LENGTH} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup-ratio ${WARMUP_RATIO} \
    --min-reward ${MIN_REWARD} \
    --deepspeed "${DS_CONFIG}"

echo "Training complete! Model saved to: ${OUTPUT_DIR}"

