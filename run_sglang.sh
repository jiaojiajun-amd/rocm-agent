docker run -it \
	--device /dev/dri \
	--device /dev/kfd \
	--network host \
	--ipc host \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
    -v /data/jiajjiao/rocm-agent:/home/jiajjiao/rocm-agent \
    --name sglang-rocm \
	-w /home/jiajjiao/rocm-agent \
	docker.io/rocm/sgl-dev:v0.5.5-rocm700-mi30x-20251108