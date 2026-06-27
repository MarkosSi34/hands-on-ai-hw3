import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from src.common import (
    MODEL_NAME, DEVICE, load_eval_corpus, perplexity, measure_size_mb,
    measure_throughput, reset_peak_memory, read_peak_memory_mb, free,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Fixed prompt for the throughput measurement (same across all configs).
THROUGHPUT_PROMPT = "The history of artificial intelligence began"


class PrecisionBenchmark:
    """
    Benchmarks one model at four precision levels and fills the Task 1 table.

    Each configuration is loaded, measured, then released before the next is
    loaded — so peak-memory numbers are not contaminated by a previous config
    still resident in (V)RAM.

    Configurations
    ──────────────
    FP32  — torch_dtype=torch.float32
    BF16  — torch_dtype=torch.bfloat16
    INT8  — load_in_8bit=True   (bitsandbytes)        [GPU only]
    INT4  — load_in_4bit=True   (bitsandbytes NF4)    [GPU only]
    """

    CONFIGS = ["FP32", "BF16", "INT8", "INT4"]
    METRICS = ["Size (MB)", "Memory (MB)", "Tokens/sec", "Perplexity"]

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Fix the evaluation corpus ONCE — reused for every configuration.
        self.corpus = load_eval_corpus(self.tokenizer)
        self.results: dict[str, dict] = {}

    # Loading one configuration
    def _load(self, config: str):
        """Load the model for a given precision config. Returns the model."""
        logging.info(f"Loading model at {config} precision...")
        if config == "FP32":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32).to(DEVICE)
        if config == "BF16":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16).to(DEVICE)
        if config == "INT8":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        if config == "INT4":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16))
        raise ValueError(f"Unknown config: {config}")

    # Measuring one configuration
    def _benchmark_one(self, config: str):
        reset_peak_memory()
        model = self._load(config)

        size_mb = measure_size_mb(model)
        ppl = perplexity(model, self.corpus)
        tok_s = measure_throughput(model, self.tokenizer, THROUGHPUT_PROMPT)
        # Peak memory: GPU via torch; on CPU fall back to the footprint as a
        # proxy (tracemalloc only tracks Python allocations, not torch tensors).
        peak = read_peak_memory_mb()
        if peak is None:
            peak = size_mb  # CPU proxy — note this in the README.

        row = {
            "Size (MB)":   round(size_mb, 1),
            "Memory (MB)": round(peak, 1),
            "Tokens/sec":  round(tok_s, 1),
            "Perplexity":  round(ppl, 3),
        }
        logging.info(f"{config}: {row}")
        free(model)
        return row

    # Output
    def _print_table(self):
        sep = "-" * 64
        logging.info(sep)
        logging.info("TASK 1 — PRECISION BENCHMARK")
        logging.info(sep)
        header = f"{'Precision':<10}" + "".join(f"{m:>14}" for m in self.METRICS)
        logging.info(header)
        logging.info(sep)
        for cfg in self.CONFIGS:
            if cfg not in self.results:
                continue
            r = self.results[cfg]
            logging.info(f"{cfg:<10}" + "".join(f"{r[m]:>14}" for m in self.METRICS))
        logging.info(sep)

    def _plot(self, path: str = "results/task1_benchmarks.png"):
        os.makedirs("results", exist_ok=True)
        configs = [c for c in self.CONFIGS if c in self.results]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Task 1 — Precision vs Size / Memory / Speed / Quality",
                     fontsize=14, fontweight="bold")

        for ax, metric in zip(axes.flat, self.METRICS):
            vals = [self.results[c][metric] for c in configs]
            bars = ax.bar(configs, vals, color="steelblue", alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{bar.get_height():g}", ha="center", va="bottom", fontsize=9)
            ax.set_title(metric, fontsize=12)
            ax.set_ylabel(metric)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logging.info(f"Saved '{path}'")

    # Orchestration
    def run_pipeline(self, configs: list[str] | None = None):
        configs = configs or self.CONFIGS
        for cfg in configs:
            try:
                self.results[cfg] = self._benchmark_one(cfg)
            except Exception as exc:  # e.g. bitsandbytes INT8/INT4 need a GPU
                logging.warning(f"Skipping {cfg}: {exc}")
        self._print_table()
        self._plot()
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Task 1: Precision Benchmarking")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="Model id (default: the project model in src/common.py).")
    parser.add_argument("--configs", nargs="+", default=None,
                        choices=PrecisionBenchmark.CONFIGS,
                        help="Subset of precision configs to run (default: all).")
    args = parser.parse_args()

    PrecisionBenchmark(model_name=args.model).run_pipeline(args.configs)


if __name__ == "__main__":
    main()
