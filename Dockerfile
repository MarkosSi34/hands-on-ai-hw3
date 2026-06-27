# HW3 — reproducible GPU image. Bundles the EXACT environment (which was painful
# to assemble: transformers 5.x + gptqmodel + hqq + the dataset swaps). Models are
# NOT baked into the image — they download at runtime and cache to a mounted volume.
#
# Build:
#   docker build -t hw3-quant .
#
# Run (needs nvidia-container-toolkit). Pick which task(s); mount the HF cache so
# models/datasets are not re-downloaded, and results/ so plots land on the host:
#   docker run --gpus all -e GPTQ_BACKEND=auto \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $(pwd)/results:/app/results \
#     hw3-quant --tasks all
#   docker run --gpus all hw3-quant --tasks 1            # just Task 1 (fills INT8)
#   docker run --gpus all hw3-quant --tasks 1 2 --model Qwen/Qwen2.5-1.5B   # scaling
#
# The CUDA *devel* base ships nvcc, so the fast Marlin GPTQ kernel can JIT-compile
# (CUDA_HOME is set) — exactly what the GTX 1660 dev box could not do.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    GPTQ_BACKEND=auto

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# torch/torchvision first, from the CUDA-matched wheel index. Adjust the index
# (cu124 / cu126 / cu128) to whatever build of torch==2.12.1 is published for your
# host's CUDA — the rest of the stack is CUDA-agnostic.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        torch==2.12.1 torchvision==0.27.1 \
        --index-url https://download.pytorch.org/whl/cu124

# the rest of the pinned stack (torch already satisfied → not reinstalled)
COPY requirements.lock.txt .
RUN python -m pip install --no-cache-dir -r requirements.lock.txt

COPY . .

ENTRYPOINT ["python", "run_all.py"]
CMD ["--tasks", "all"]
