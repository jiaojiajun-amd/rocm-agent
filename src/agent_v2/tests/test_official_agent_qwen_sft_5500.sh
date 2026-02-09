python test_rocm_agent_amd.py test_all_multi_thread \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "jsonjiao/qwen3-rocm-sft-5500step" \
    --docker-server "10.235.85.27:9527" \
    --eval-server "10.235.85.27:9528" \
    --output /home/jiajjiao/rocm-agent/results/rocprimv5_official_agent_qwen_sft_5500_r1.json \
    --simple-output /home/jiajjiao/rocm-agent/results/rocprimv5_official_agent_qwen_sft_5500_r1_simple.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --use-rocm-agent mini \
    --workers 16

