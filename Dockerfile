FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

LABEL maintainer david.lindenbaum@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy

RUN mkdir -p /home/spacenet
ENV SPACENET_ROOT=/home/spacenet
WORKDIR $SPACENET_ROOT

#RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/dlindenbaum/spaceTest.git . && \
    sudo cp -R sources.list* /etc/apt/
    sudo apt-get update
    sudo apt-get install dselect
    sudo dselect update
    sudo dpkg --set-selections < Package.list
    sudo apt-get dselect-upgrade -y
    pip install --upgrade pip && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. 
