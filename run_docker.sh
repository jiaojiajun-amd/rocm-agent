docker run -it \
	--device /dev/dri \
	--device /dev/kfd \
	--network host \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
    -v $HOME/code/mini-swe-agent:/home/jiajjiao/code/mini-swe-agent \
	--shm-size 128G \
	-w /home/jiajjiao/rllm \
    --name swe-verl-1 \
	jiajjiao/verl-rocm:0.5.0