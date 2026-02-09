# python test_rocm_agent_amd.py test-single \
#     --instance /home/jiajjiao/rocm-agent/data/device_merge_sort.json \
#     --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
#     --model "gpt-5" \
#     --docker-server "10.67.77.184:9527" \
#     --eval-server "10.67.77.184:9528" \
#     --output /home/jiajjiao/rocm-agent/results/single_result.json

# python test_rocm_agent_amd.py test-all \
#     --dataset /home/jiajjiao/rocm-agent/data/rocprim_v2.json \
#     --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
#     --model "gpt-5" \
#     --docker-server "10.67.77.184:9527" \
#     --eval-server "10.67.77.184:9528" \
#     --output /home/jiajjiao/rocm-agent/results/all_results.json \

python test_rocm_agent_amd.py test_all_multi_thread \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v5.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "zai-org/GLM-4.6" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/results/all_results_rocprim_v5_glm46_w_mem.json \
    --config /home/jiajjiao/rocm-agent/src/minisweagent/config/rocm/config_amd.yaml \
    --workers 16



