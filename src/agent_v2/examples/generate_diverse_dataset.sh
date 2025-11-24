#!/bin/bash
# Generate a diverse dataset with higher temperature for exploration

python ../generate_training_data.py generate-multi \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "Qwen/Qwen3-8B" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/training_data/diverse_dataset.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 4 \
    --temperature 1.5 \
    --max-tokens 8000 \
    --samples-per-task 32 \
    --log-file /home/jiajjiao/rocm-agent/training_data/diverse_dataset.log

