#!/bin/bash
# Single GPU CPT training script for Qwen3-8B (for debugging/testing)
# Usage: bash run_cpt_single_gpu.sh

set -e

# Configuration
DATA_FILE="${DATA_FILE:-/home/jiajjiao/rocm-agent/training_data/large_dataset.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jiajjiao/rocm-agent/outputs/qwen3-8b-cpt-debug}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"

# Training hyperparameters (smaller for single GPU)
MAX_LENGTH="${MAX_LENGTH:-8192}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
MIN_REWARD="${MIN_REWARD:-0.5}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPT_SCRIPT="${SCRIPT_DIR}/../cpt_qwen3_8b.py"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Single GPU CPT Training for Qwen3-8B"
echo "=========================================="
echo "Data file: ${DATA_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model: ${MODEL}"
echo "Max length: ${MAX_LENGTH}"
echo "Epochs: ${EPOCHS}"
echo "Min reward: ${MIN_REWARD}"
echo "=========================================="

# Run training
CUDA_VISIBLE_DEVICES=0 python "${CPT_SCRIPT}" \
    --data-file "${DATA_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --max-length ${MAX_LENGTH} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --min-reward ${MIN_REWARD}

echo "Training complete! Model saved to: ${OUTPUT_DIR}"

