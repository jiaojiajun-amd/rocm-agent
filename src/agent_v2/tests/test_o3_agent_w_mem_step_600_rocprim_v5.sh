python test_rocm_agent_amd.py test_all_multi_thread \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "o3" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/results/all_results_o3_v5_resoning_w_mem_step_200.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/rocm/config_amd.yaml \
    --workers 8

