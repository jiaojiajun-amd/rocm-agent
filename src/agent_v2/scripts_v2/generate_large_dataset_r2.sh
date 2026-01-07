#!/bin/bash
# Generate SFT training data - Round 2 (v2)
# Output:
#   - SFT data: training_data/sft_large_dataset_r2.json
#   - Trajectories: training_data/sft_large_dataset_r2_trajectories/

python /home/jiajjiao/rocm-agent/src/agent_v2/generate_training_data.py \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "Claude-Sonnet-4.5" \
    --docker-server "10.235.85.27:9527" \
    --eval-server "10.235.85.27:9528" \
    --output /home/jiajjiao/rocm-agent/training_data_v2/sft_large_dataset_r2.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 8 \
    --temperature 1.0 \
    --repeats 8

