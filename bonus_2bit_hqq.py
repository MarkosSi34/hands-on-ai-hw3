"""
Bonus A — 2-bit Quantization with HQQ.

Extends the Task 1 benchmark with an INT2 (HQQ) row and plots the full
precision curve FP32 -> BF16 -> INT8 -> INT4 -> INT2, identifying the quality
floor (the point where perplexity degrades sharply).

HQQ (Half-Quadratic Quantization) needs NO calibration data, supports 2-bit
natively, and runs on CPU and consumer GPUs.

Run:
    python bonus_2bit_hqq.py

Produces:
    results/bonus_2bit_precision_curve.png
"""
import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hqq.core.quantize import BaseQuantizeConfig
from hqq.models.hf.base import AutoHQQHFModel

from src.common import (
    MODEL_NAME, DEVICE, load_eval_corpus, perplexity, measure_size_mb,
    measure_throughput, reset_peak_memory, read_peak_memory_mb, free,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

THROUGHPUT_PROMPT = "The history of artificial intelligence began"


class HQQ2BitBenchmark:
    """Quantizes the model to 2-bit with HQQ and measures the full Task-1 metric
    set (size, peak memory, throughput, perplexity) for the INT2 row."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.corpus = load_eval_corpus(self.tokenizer)

    def _load_hqq_2bit(self):
        # transformers 5.x has not wired up the HqqConfig loading path yet
        # ("QuantizationMethod.HQQ is not available yet"), so we use HQQ's own
        # native API: load an FP16 model, then quantize it in-place.
        # nbits=2, group_size=64 is the standard aggressive 2-bit setting.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16)
        quant_config = BaseQuantizeConfig(nbits=2, group_size=64)
        AutoHQQHFModel.quantize_model(
            model, quant_config=quant_config,
            compute_dtype=torch.float16, device=str(DEVICE))
        return model

    def run_pipeline(self):
        reset_peak_memory()
        model = self._load_hqq_2bit()
        size = measure_size_mb(model)
        ppl = perplexity(model, self.corpus)
        tok_s = measure_throughput(model, self.tokenizer, THROUGHPUT_PROMPT)
        peak = read_peak_memory_mb()          # None on CPU
        free(model)
        row = {
            "Size (MB)":   round(size, 1),
            "Memory (MB)": round(peak, 1) if peak is not None else None,
            "Tokens/sec":  round(tok_s, 1),
            "Perplexity":  round(ppl, 3),
        }
        logging.info(f"INT2 (HQQ): {row}")
        return row


def plot_precision_curve(points: dict[str, tuple[float, float]],
                         path: str = "results/bonus_2bit_precision_curve.png"):
    """
    points: {label: (size_mb, perplexity)} in increasing-compression order,
            e.g. FP32 -> BF16 -> INT8 -> INT4 -> INT2.
    Plots perplexity vs model size so the quality floor is visible.
    """
    os.makedirs("results", exist_ok=True)
    labels = list(points.keys())
    sizes = [points[l][0] for l in labels]
    ppls = [points[l][1] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sizes, ppls, "o-", color="darkred", lw=2)
    for label, s, p in zip(labels, sizes, ppls):
        ax.annotate(label, (s, p), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Model size on disk (MB)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Bonus A — Precision curve FP32 → INT2 (quality floor)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Saved '{path}'")


def main():
    parser = argparse.ArgumentParser(description="Bonus A — 2-bit HQQ")
    parser.parse_args()

    hqq_row = HQQ2BitBenchmark().run_pipeline()
    points = {
        "FP32": (1884.6, 22.733),
        "BF16": (942.3, 22.664),
        "INT8": (601.0, 22.936),
        "INT4": (430.4, 27.502),
        "INT2": (hqq_row["Size (MB)"], hqq_row["Perplexity"]),
    }
    plot_precision_curve(points)


if __name__ == "__main__":
    main()
