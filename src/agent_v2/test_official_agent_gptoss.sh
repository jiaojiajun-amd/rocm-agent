python test_rocm_agent_amd.py test_all_multi_thread \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "lmsys/gpt-oss-20b-bf16" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/results/rocprimv5_official_agent_gpt_oss.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/mini.yaml \
    --use-rocm-agent mini \
    --workers 16

