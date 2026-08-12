FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        software-properties-common \
        wget \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        g++ \
        g++-13 \
        gcc \
        gcc-13 \
        gdb \
        gfortran \
        gfortran-13 \
        git \
        hwloc \
        libgomp1 \
        make \
        netcat-openbsd \
        openssh-client \
        pkg-config \
        python3 \
        python3-matplotlib \
        python3-numpy \
        python3-pip \
        python3-venv \
        rsync \
        zlib1g-dev \
        intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

RUN if [[ ! -d /opt/intel/oneapi/mkl/2025.0 && -d /opt/intel/oneapi/mkl/latest ]]; then \
        ln -sfn /opt/intel/oneapi/mkl/latest /opt/intel/oneapi/mkl/2025.0; \
    fi

COPY docker/build-mpich-and-petsc.sh /tmp/build-mpich-and-petsc.sh
RUN chmod +x /tmp/build-mpich-and-petsc.sh \
    && BUILD_JOBS="${BUILD_JOBS:-$(nproc)}" /tmp/build-mpich-and-petsc.sh \
    && rm -rf /tmp/build-mpich-and-petsc.sh /home/votrongnam/src

COPY docker/env.sh /opt/large-scale-to/env.sh
COPY docker/entrypoint.sh /opt/large-scale-to/entrypoint.sh
RUN chmod +x /opt/large-scale-to/entrypoint.sh

ENV TO_TOOLCHAIN=gcc13
ENV MKLROOT=/opt/intel/oneapi/mkl/2025.0
WORKDIR /workspace/Large_scale_TO

ENTRYPOINT ["/opt/large-scale-to/entrypoint.sh"]
CMD ["bash"]
