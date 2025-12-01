#!/bin/bash
# Resume large dataset generation for first 29 tasks of rocprim_v5
# With reduced concurrency (2 workers) and resume support
# Only counts successful samples, skips already complete tasks

python ../generate_training_data.py generate-multi \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5_resume.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "Claude-Sonnet-4.5" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/training_data/large_dataset.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 4 \
    --temperature 1.0 \
    --samples-per-task 32 \
    --resume \
    --log-file /home/jiajjiao/rocm-agent/training_data/large_dataset.log



