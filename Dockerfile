
FROM nvidia/cuda:8.0-devel-ubuntu14.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
ENV CUDNN_VERSION 5.1.10
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn5=$CUDNN_VERSION-1+cuda8.0 \
            libcudnn5-dev=$CUDNN_VERSION-1+cuda8.0 && \

ENV SPACENET_ROOT=/home/spacenet
WORKDIR $SPACENET_ROOT

RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/dlindenbaum/spaceTest.git . && \
    sudo cp -R sources.list* /etc/apt/
    sudo apt-get update
    sudo apt-get install dselect
    sudo dselect update
    sudo dpkg --set-selections < Package.list
    sudo apt-get dselect-upgrade -y
    pip install --upgrade pip && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. 
