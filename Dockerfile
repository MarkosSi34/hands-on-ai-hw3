# HW3 — reproducible GPU image (same self-contained style as the HW1 image:
# clone-from-GitHub + venv). Unlike HW1, HW3 needs a CUDA *devel* base so that
# nvcc is present and the fast Marlin GPTQ kernel can JIT-compile — that is what
# lets this image "run everything" (INT8 + Marlin) that the GTX 1660 could not.
#
# Models are NOT baked into the image: they download at runtime and cache to a
# mounted volume.
#
# Build:
#   docker build -t hw3-quant .
#
# Run (needs nvidia-container-toolkit). Mount the HF cache so models/datasets are
# not re-downloaded, and results/ so the plots land on the host:
#   docker run --gpus all \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $(pwd)/results:/app/results \
#     hw3-quant --tasks all
#   docker run --gpus all hw3-quant --tasks 1            # just Task 1 (fills INT8)
#   docker run --gpus all hw3-quant --tasks 1 2 --model Qwen/Qwen2.5-1.5B   # scaling
#
# For a GATED model (e.g. meta-llama/Llama-3.2-1B-Instruct) pass an HF token:
#   docker run --gpus all -e HF_TOKEN=hf_xxx \
#     hw3-quant --tasks 1 2 3 bonus --model meta-llama/Llama-3.2-1B-Instruct

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    GPTQ_BACKEND=auto

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Self-contained: pull the code straight from GitHub (same as the HW1 image).
RUN git clone -b gpu-branch https://github.com/MarkosSi34/hands-on-ai-hw3.git .

# venv, then torch/torchvision FIRST, pinned to the lockfile versions and pulled
# from the CUDA-matched wheel index. Because the version matches the lockfile, the
# `-r requirements.lock.txt` step below sees torch as already satisfied and does
# NOT re-pull a CPU build. The cu124 wheel ships its own CUDA 12.4 runtime, which
# runs fine under this image's newer system CUDA (forward-compatible driver).
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir \
        torch==2.12.1 torchvision==0.27.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && .venv/bin/pip install --no-cache-dir -r requirements.lock.txt

ENTRYPOINT [".venv/bin/python", "run_all.py"]
CMD ["--tasks", "all"]
