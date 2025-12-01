#!/bin/bash
# Retry failed tasks with reduced concurrency (2 workers instead of 8)
# This should be more stable and avoid 500 Server Errors

python ../generate_training_data.py generate-multi \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5_h2_retry.json \
    --api-key "65481c6913004dafb08eee42587844c0" \
    --model "Claude-Sonnet-4.5" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/training_data/large_dataset_h2_retry.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 4 \
    --temperature 1.0 \
    --samples-per-task 32 \
    --log-file /home/jiajjiao/rocm-agent/training_data/large_dataset_h2_retry.log



