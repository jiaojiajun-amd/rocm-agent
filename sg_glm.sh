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
#   --model-path zai-org/GLM-4.5-Air \
#   --tp 8 \
#   --dtype bfloat16 \
#   --trust-remote-code \

# HIP_VISIBLE_DEVICES=4,5,6,7

 SGLANG_USE_AITER=0 python3 -m sglang.launch_server --model zai-org/GLM-4.5-Air --attention-backend triton --tp 4 --trust-remote-code 



