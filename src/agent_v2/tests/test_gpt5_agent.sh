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

python test_rocm_agent_amd.py test-all \
    --dataset /home/jiajjiao/rocm-agent/data/rocprim_v4.json \
    --api-key "c1f7f3ee59064fc0a5fad8c2586f1bd9" \
    --model "gpt-5" \
    --docker-server "10.67.77.184:9527" \
    --eval-server "10.67.77.184:9528" \
    --output /home/jiajjiao/rocm-agent/results/all_results_v4_resoning_high_config_v1_repeat_1.json \
    --workers 8


