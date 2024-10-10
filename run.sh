docker run --rm -it \
	-v /home/ubuntu/vista2d_rapids_clustering:/workspace \
	-w /workspace \
	-p 8888:8888 \
	--gpus all \
	projectmonai/monai:1.4.0rc10 \
	/bin/bash
