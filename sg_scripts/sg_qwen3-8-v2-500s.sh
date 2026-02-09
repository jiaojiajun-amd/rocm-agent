# export SGLANG_SKIP_MARLIN_REPACK=1
# export USE_ROCM_AITER_ROPE_BACKEND=0
# export SGLANG_DISABLE_MARLIN_MOE=1
# export SGLANG_DISABLE_COMPRESSED_TENSORS=1


# export HSA_FORCE_FINE_GRAIN_PCIE=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand
# export NCCL_SOCKET_IFNAME=enp0s20f0u10u4
# export GLOO_SOCKET_IFNAME=enp0s20f0u10u4
# export SGLANG_USE_AITER=0

# python -m sglang.launch_server \
#   --model-path Qwen/Qwen3-8B \
#   --tp 1 \
#   --dtype bfloat16 \
#   --trust-remote-code \
#   --disable-cuda-graph

# # python -m sglang.launch_server \
# #   --model-path Qwen/Qwen3-8B \
# #   --tp 1 \
# #   --dtype bfloat16 \
# #   --trust-remote-code \
# #   --cuda-graph-max-bs 32

HIP_VISIBLE_DEVICES=0 SGLANG_USE_AITER=0 python3 -m sglang.launch_server --model jsonjiao/qwen3-rocm-sft-v2-500step --attention-backend triton --tp 1 --trust-remote-code --port 30000 --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' --context-length 131072
