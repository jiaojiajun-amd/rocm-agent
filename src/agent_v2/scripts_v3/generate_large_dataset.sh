#!/bin/bash
# Generate SFT training data (v3 - with auto history summarization)
# Uses max_context_tokens=24000 in mini.yaml for automatic context management
# Output:
#   - SFT data: training_data_v3/sft_large_dataset.json
#   - Trajectories: training_data_v3/sft_large_dataset_trajectories/

python /home/jiajjiao/rocm-agent/src/agent_v2/generate_training_data.py \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "65481c6913004dafb08eee42587844c0" \
    --model "Claude-Sonnet-4.5" \
    --docker-server "10.235.85.27:9527" \
    --eval-server "10.235.85.27:9528" \
    --output /home/jiajjiao/rocm-agent/training_data_v3/sft_large_dataset.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 4 \
    --temperature 1.0 \
    --repeats 4

