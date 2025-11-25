#!/bin/bash
# Generate a large dataset for production training (all tasks, 8 workers)

python ../generate_training_data.py generate-multi \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5_h2.json \
    --api-key "65481c6913004dafb08eee42587844c0" \
    --model "Claude-Sonnet-4.5" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/training_data/large_dataset_h2.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 8 \
    --temperature 1.0 \
    --samples-per-task 32 \
    --log-file /home/jiajjiao/rocm-agent/training_data/large_dataset_h2.log

