docker run -it \
	--device /dev/dri \
	--device /dev/kfd \
	--network host \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
	--shm-size 128G \
	-w /app \
    --name swe-lib-test \
	rocm-lib