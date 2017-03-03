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
        python-scipy \
        python-gdal \
        gdal-bin \
        libopencv-dev \
        python-opencv

RUN mkdir -p /home/spacenet
ENV SPACENET_ROOT=/home/spacenet
WORKDIR $SPACENET_ROOT

#RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py
RUN git clone --depth 1 https://github.com/dlindenbaum/spaceTest.git .


RUN sudo cp -R sources.list /etc/apt/
RUN sudo apt-get update
#RUN sudo apt-get install dselect
#RUN sudo dselect update
#RUN sudo dpkg --set-selections < Package.list
#RUN sudo apt-get dselect-upgrade -y
#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
RUN apt-get -y install clang
RUN export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
RUN sudo pip install --upgrade $TF_BINARY_URL
RUN ./compile.sh


RUN sudo pip install bunch pandas
