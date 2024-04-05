# Use official Ubuntu image as a parent image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables to suppress warnings and to avoid prompts
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn \
    DEBIAN_FRONTEND=noninteractive

# Install packages and clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    wget git build-essential cmake libopenmpi-dev python3-dev zlib1g-dev \
    libgl1-mesa-glx swig libopenblas-dev libsuitesparse-dev libgsl0-dev \
    libfftw3-dev libglpk-dev libdsdp-dev python3.10 python3-pip nano \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Clone your training-farm repository and install additional requirements
RUN git clone https://github.com/mikazlopes/d2drl.git && \
    cd d2drl && \
    pip install -r requirements.txt && \
    cd dreamerv3 && \
    pip install -r requirements.txt

EXPOSE 8120-8140

WORKDIR /d2drl