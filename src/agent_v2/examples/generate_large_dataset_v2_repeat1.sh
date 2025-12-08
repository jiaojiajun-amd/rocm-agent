#!/bin/bash
# Generate a large dataset for production training (reduced to 2 workers to avoid 500 errors)
# DEPRECATED: Use generate_large_dataset_resume.sh with --resume flag instead

python ../generate_training_data.py generate-multi \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "Claude-Sonnet-4.5" \
    --docker-server "10.235.85.27:9527" \
    --eval-server "10.235.85.27:9528" \
    --output /home/jiajjiao/rocm-agent/training_data/large_dataset_v2_repeat1.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 1 \
    --temperature 1.0 \
    --samples-per-task 1 \

