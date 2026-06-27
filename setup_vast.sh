#!/usr/bin/env bash
# setup_vast.sh — προετοιμασία ενός vast.ai instance (ή οποιουδήποτε box που έχει
# ΗΔΗ torch + CUDA, π.χ. ένα PyTorch template) για να τρέξει το HW3 pipeline.
#
# ΓΙΑΤΙ δεν κάνουμε `pip install -r requirements.lock.txt`: το lock καρφώνει
# torch==2.12.1, που θα ΑΝΤΙΚΑΘΙΣΤΟΥΣΕ το (σωστά CUDA-matched) torch του image και
# θα χαλούσε το CUDA. Εδώ αφήνουμε το torch/torchvision/triton του image ως έχουν
# και εγκαθιστούμε μόνο το υπόλοιπο stack.
#
# Χρήση (μέσα στο instance):
#   bash setup_vast.sh
#   GPTQ_BACKEND=auto python run_all.py --tasks all
set -euo pipefail

echo "== Υπάρχον torch / CUDA (ΔΕΝ το πειράζουμε) =="
python -c "import torch; print('torch', torch.__version__, '| cuda?', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

echo "== Εγκατάσταση του stack (χωρίς torch/torchvision) =="
# Σταθερές εκδόσεις όπου έχει σημασία· τα torch-coupled (torchao/triton/bitsandbytes)
# αφήνονται να ταιριάξουν με το torch του image.
pip install --no-cache-dir \
    "transformers==5.12.1" \
    "tokenizers==0.22.2" \
    "accelerate==1.14.0" \
    "datasets==5.0.0" \
    "huggingface-hub==1.21.0" \
    "safetensors==0.8.0" \
    "numpy==2.2.6" \
    "matplotlib==3.11.0" \
    "optimum" \
    "bitsandbytes" \
    "gptqmodel" \
    "hqq" \
    "sentencepiece" \
    "protobuf"

echo "== Έλεγχος imports =="
python -c "import transformers, datasets, bitsandbytes, gptqmodel, hqq; print('stack OK:', transformers.__version__)"

echo
echo "Έτοιμο. Τρέξε π.χ.:"
echo "  GPTQ_BACKEND=auto python run_all.py --tasks all"
echo "  GPTQ_BACKEND=auto python run_all.py --tasks 1 2 --model Qwen/Qwen2.5-1.5B"
