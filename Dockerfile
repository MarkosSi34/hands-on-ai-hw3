FROM docker.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    GPTQ_BACKEND=auto

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Self-contained: pull the code straight from GitHub (same as the HW1 image).
RUN git clone https://github.com/MarkosSi34/hands-on-ai-hw3.git .

# venv, then torch/torchvision FIRST, pinned to the lockfile versions and pulled
# from the CUDA-matched wheel index. Because the version matches the lockfile, the
# `-r requirements.lock.txt` step below sees torch as already satisfied and does
# NOT re-pull a CPU build. torch 2.12.1 is a cu130 build (matches the RTX 4090 run
# stack: CUDA 13.1 + torch 2.12.x+cu130), so we use the cu130 wheel index and a
# CUDA 13 devel base — nvcc and the torch runtime are then on the same CUDA major.
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir \
        torch==2.12.1 torchvision==0.27.1 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && .venv/bin/pip install --no-cache-dir -r requirements.lock.txt

ENTRYPOINT [".venv/bin/python", "run_all.py"]
CMD ["--tasks", "all"]
