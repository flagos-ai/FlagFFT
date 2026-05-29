FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_URL_REWRITE_FROM=https://github.com/
ARG GIT_URL_REWRITE_TO=

SHELL ["/bin/bash", "-lc"]

ENV PYENV_ROOT=/root/.pyenv
ENV PATH=${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "${GIT_URL_REWRITE_TO}" ]; then \
        git config --global url."${GIT_URL_REWRITE_TO}".insteadOf "${GIT_URL_REWRITE_FROM}"; \
    fi

# pyenv
RUN curl -fsSL https://pyenv.run | bash

RUN pyenv install 3.12 && \
    pyenv global 3.12 && \
    python --version && \
    pip --version

# torch
RUN pip3 install --no-warn-conflicts torch torchvision --index-url https://download.pytorch.org/whl/cu132

# cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda-repo-debian13-13-2-local_13.2.1-595.58.03-1_amd64.deb && \
    dpkg -i cuda-repo-debian13-13-2-local_13.2.1-595.58.03-1_amd64.deb && \
    cp /var/cuda-repo-debian13-13-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-13-2 && \
    rm -f cuda-repo-debian13-13-2-local_13.2.1-595.58.03-1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# flagtree
RUN pip uninstall -y triton || true && \
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple" && \
    pip install --no-warn-conflicts flagtree===0.5.0 ${RES}

# flagFFT build dependencies
RUN pip install --no-warn-conflicts nanobind ninja cmake pybind11 && \
    pip install --no-warn-conflicts -U pip "setuptools<82" wheel scikit-build-core

WORKDIR /workspace

CMD ["/bin/bash"]
