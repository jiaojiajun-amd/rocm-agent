docker run -it \
	--device /dev/dri \
	--device /dev/kfd \
	--network host \
	--ipc host \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
    -v /home/jiajjiao/rocm-agent:/home/jiajjiao/rocm-agent \
	-v /mnt/raid0/jiajjiao/models:/models \
    --name sglang-rocm \
	docker.io/rocm/sgl-dev:v0.5.5-rocm700-mi30x-20251108