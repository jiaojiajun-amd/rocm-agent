#!/bin/bash
# Complete data processing pipeline for training data

# Configuration
INPUT_FILE="${1:-/home/jiajjiao/rocm-agent/training_data/mini_agent_training_data.json}"
OUTPUT_DIR="${2:-/home/jiajjiao/rocm-agent/training_data/processed}"
MIN_REWARD="${3:-0.7}"

echo "===================================================="
echo "Training Data Processing Pipeline"
echo "===================================================="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Minimum reward threshold: $MIN_REWARD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Analyze raw data
echo "[1/5] Analyzing raw training data..."
python process_training_data.py analyze \
    --input "$INPUT_FILE"

echo ""

# Step 2: Filter high-quality examples
echo "[2/5] Filtering high-quality examples (reward >= $MIN_REWARD)..."
python process_training_data.py filter_data \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR/filtered_data.json" \
    --min-reward "$MIN_REWARD" \
    --successful-only

echo ""

# Step 3: Export for supervised fine-tuning
echo "[3/5] Exporting SFT format..."
python process_training_data.py export_sft \
    --input "$OUTPUT_DIR/filtered_data.json" \
    --output "$OUTPUT_DIR/sft_training_data.jsonl" \
    --min-reward "$MIN_REWARD"

echo ""

# Step 4: Export trajectory format
echo "[4/5] Exporting trajectory format..."
python process_training_data.py export_trajectory \
    --input "$OUTPUT_DIR/filtered_data.json" \
    --output "$OUTPUT_DIR/trajectory_training_data.json" \
    --min-reward "$MIN_REWARD"

echo ""

# Step 5: Analyze filtered data
echo "[5/5] Analyzing filtered data..."
python process_training_data.py analyze \
    --input "$OUTPUT_DIR/filtered_data.json"

echo ""
echo "===================================================="
echo "Pipeline completed!"
echo "===================================================="
echo "Output files:"
echo "  - Filtered data: $OUTPUT_DIR/filtered_data.json"
echo "  - SFT format: $OUTPUT_DIR/sft_training_data.jsonl"
echo "  - Trajectory format: $OUTPUT_DIR/trajectory_training_data.json"
echo ""

