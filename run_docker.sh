docker run -it \
	--device /dev/dri \
	--device /dev/kfd \
	--network host \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
    -v $HOME/rocm-agent/agent:/home/jiajjiao/rocm-agent/agent \
	--shm-size 128G \
	-w /app \
    --name swe-verl-1 \
	rocm-rl