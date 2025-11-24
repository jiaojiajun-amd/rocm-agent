#!/bin/bash
# Generate a small dataset for testing (10 tasks, 2 workers)

python ../generate_training_data.py generate-multi \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "gpt-5.1-codex" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/training_data/small_dataset.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --workers 8 \
    --max-tasks 1 \
    --samples-per-task 32 \
    --log-file /home/jiajjiao/rocm-agent/training_data/small_dataset.log

# python ../generate_training_data.py generate-multi \
#     --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
#     --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
#     --model "zai-org/GLM-4.5-Air" \
#     --docker-server "10.67.77.184:9527" \
#     --eval-server "10.67.77.184:9528" \
#     --output /home/jiajjiao/rocm-agent/training_data/small_dataset-glm.json \
#     --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
#     --workers 16 \
#     --max-tasks 1 \
#     --samples-per-task 32 \
#     --log-file /home/jiajjiao/rocm-agent/training_data/small_dataset_glm.log

