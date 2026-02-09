python test_rocm_agent_amd.py test_all_multi_thread \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" --model "Claude-Opus-4.5" \
    --docker-server "10.235.85.27:9527" \
    --eval-server "10.235.85.27:9528" \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini_debug.yaml \
    --use-rocm-agent mini --workers 1 \
    --output-dir /home/jiajjiao/rocm-agent/results_debug