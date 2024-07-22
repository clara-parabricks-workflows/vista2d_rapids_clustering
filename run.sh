docker run --rm -it \
	-v /home/tongz/gburnett/vista2d:/workspace \
	-p 8888:8888 \
	--gpus all \
	nvcr.io/nvidia/pytorch:24.03-py3 /bin/bash
