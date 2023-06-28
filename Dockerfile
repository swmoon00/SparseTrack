FROM nvcr.io/nvidia/pytorch:23.04-py3

# install packages
RUN apt update && apt install -y libboost-python-dev libpython3-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt install -y cmake libopencv-dev

# clone repo
RUN git clone https://github.com/swmoon00/SparseTrack.git
RUN git clone https://github.com/Algomorph/pyboostcvconverter.git

# make substitutions
RUN sed -i "s/\/home\/algomorph\/Factory/\/workspace/g" pyboostcvconverter/Makefile
RUN cp SparseTrack/python_module.cpp pyboostcvconverter/src/
RUN sed -i "s/core/core highgui video videoio videostab/" pyboostcvconverter/CMakeLists.txt

RUN (export CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake && cd pyboostcvconverter && make)
RUN cp pyboostcvconverter/pbcvt.cpython-38-x86_64-linux-gnu.so SparseTrack/tracker

WORKDIR /workspace/SparseTrack
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/swmoon00/cython_bbox.git