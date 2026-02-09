#!/bin/bash
# Quick test script for training data generation with mini agent

echo "Testing training data generation system..."
echo "========================================="
echo ""

# Test with a small number of tasks
python generate_training_data.py generate-multi \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "Qwen/Qwen3-8B" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/training_data/test_training_data.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 2 \
    --max-tasks 3 \
    --temperature 1.0 \
    --max-tokens 8000 \
    --samples-per-task 2

echo ""
echo "========================================="
echo "Analyzing generated data..."
echo "========================================="
echo ""

python process_training_data.py analyze \
    --input /home/jiajjiao/rocm-agent/training_data/test_training_data.json

echo ""
echo "========================================="
echo "Showing first example..."
echo "========================================="
echo ""

python process_training_data.py show_example \
    --input /home/jiajjiao/rocm-agent/training_data/test_training_data.json \
    --index 0

echo ""
echo "Test completed!"

