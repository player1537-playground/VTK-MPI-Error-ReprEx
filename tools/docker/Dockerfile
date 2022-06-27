FROM ubuntu:focal AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        gcc \
        g++ \
        gfortran \
        make \
        cmake \
        libgl1-mesa-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        gdb \
    && \
    rm -rf /var/lib/apt/lists/*
