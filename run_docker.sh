docker run -it \
	--device /dev/dri \
	--device /dev/kfd \
	--network host \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
    -v /data/jiajjiao/rocm-agent:/home/jiajjiao/rocm-agent \
	--shm-size 128G \
	-w /home/jiajjiao/rocm-agent \
    --name rocm-agent \
	rocm-rl