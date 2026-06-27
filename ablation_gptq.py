"""
Ablation — GPTQ hyperparameters vs perplexity.

Δύο sweeps στο σταθερό WikiText-2 eval corpus (ίδιο `load_eval_corpus` με τα
Tasks 1–3), ώστε να δείξουμε ότι εξερευνήσαμε τους knobs του GPTQ αντί να
χρησιμοποιήσουμε απλώς τα defaults:

  (A) group_size ∈ {32, 64, 128}      με n_calib = 128 σταθερό
  (B) n_calib    ∈ {32, 64, 128, 256} με group_size = 128 σταθερό

Calibration: WikiText-2 **train** split (όπως στο Task 2). Inference kernel:
env GPTQ_BACKEND (default TORCH· AUTO σε Ampere+).

Run:
    python ablation_gptq.py
    GPTQ_BACKEND=auto python ablation_gptq.py        # σε Ampere+ GPU

Παράγει:
    results/ablation_gptq.png
    cached models στο gptq_ablation/  (NOT committed — βλ. .gitignore)
"""
import argparse
import logging
import os

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig, BACKEND

from src.common import MODEL_NAME, DEVICE, load_eval_corpus, perplexity, free

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

GROUP_SIZES = [32, 64, 128]
N_CALIBS = [32, 64, 128, 256]
GROUP_SIZE_DEFAULT = 128
N_CALIB_DEFAULT = 128
ABLATION_DIR = "gptq_ablation"


def _backend():
    """GPTQ inference kernel from env GPTQ_BACKEND (default TORCH)."""
    return getattr(BACKEND, os.environ.get("GPTQ_BACKEND", "TORCH").upper(),
                   BACKEND.TORCH)


def _calib_texts(n: int):
    """First `n` WikiText-2 TRAIN paragraphs (≥64 chars) as calibration strings."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    out = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= 64:
            out.append(text)
        if len(out) >= n:
            break
    return out


def _build_and_eval(model_name, tokenizer, corpus, calib, group_size, n_calib):
    """Quantise (or load cached) one GPTQ config and return its WikiText PPL."""
    cache = os.path.join(ABLATION_DIR, f"g{group_size}_n{n_calib}")
    if not os.path.isdir(cache):
        logging.info(f"Calibrating GPTQ group_size={group_size}, n_calib={n_calib}...")
        cfg = QuantizeConfig(bits=4, group_size=group_size)
        model = GPTQModel.load(model_name, cfg)
        model.quantize(calib[:n_calib], tokenizer=tokenizer, batch_size=1)
        os.makedirs(ABLATION_DIR, exist_ok=True)
        model.save(cache)
        free(model)
    model = GPTQModel.load(cache, backend=_backend())
    ppl = perplexity(model, corpus)
    free(model)
    return ppl


def _report(gs_ppl, nc_ppl):
    sep = "-" * 52
    logging.info(sep)
    logging.info("ABLATION — GPTQ hyperparameters vs WikiText-2 PPL")
    logging.info(sep)
    logging.info(f"(A) group_size sweep  (n_calib={N_CALIB_DEFAULT})")
    for gs in GROUP_SIZES:
        logging.info(f"    group_size={gs:<4} -> PPL {gs_ppl[gs]:.3f}")
    logging.info(f"(B) n_calib sweep     (group_size={GROUP_SIZE_DEFAULT})")
    for nc in N_CALIBS:
        logging.info(f"    n_calib={nc:<4}    -> PPL {nc_ppl[nc]:.3f}")
    logging.info(sep)


def _plot(gs_ppl, nc_ppl, path="results/ablation_gptq.png"):
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ablation — GPTQ knobs vs WikiText-2 Perplexity",
                 fontsize=13, fontweight="bold")

    ax1.plot(GROUP_SIZES, [gs_ppl[g] for g in GROUP_SIZES], "o-", color="steelblue")
    ax1.set_title(f"group_size  (n_calib={N_CALIB_DEFAULT})")
    ax1.set_xlabel("group_size"); ax1.set_ylabel("Perplexity")
    ax1.set_xticks(GROUP_SIZES); ax1.grid(True, alpha=0.3)

    ax2.plot(N_CALIBS, [nc_ppl[n] for n in N_CALIBS], "o-", color="tomato")
    ax2.set_title(f"n_calib  (group_size={GROUP_SIZE_DEFAULT})")
    ax2.set_xlabel("n_calib samples"); ax2.set_ylabel("Perplexity")
    ax2.set_xticks(N_CALIBS); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Saved '{path}'")


def run_ablation(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    corpus = load_eval_corpus(tokenizer)
    calib_full = _calib_texts(max(N_CALIBS))

    # (A) group_size sweep — n_calib fixed
    gs_ppl = {gs: _build_and_eval(model_name, tokenizer, corpus, calib_full,
                                  gs, N_CALIB_DEFAULT) for gs in GROUP_SIZES}
    # (B) n_calib sweep — group_size fixed (g128_n128 reuses the cache from above)
    nc_ppl = {nc: _build_and_eval(model_name, tokenizer, corpus, calib_full,
                                  GROUP_SIZE_DEFAULT, nc) for nc in N_CALIBS}

    _report(gs_ppl, nc_ppl)
    _plot(gs_ppl, nc_ppl)
    return gs_ppl, nc_ppl


def main():
    parser = argparse.ArgumentParser(description="GPTQ hyperparameter ablation")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    run_ablation(args.model)


if __name__ == "__main__":
    main()
