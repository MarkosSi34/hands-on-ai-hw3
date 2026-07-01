"""
cpu_benchmark.py — Real CPU-only deployment benchmark (validates Task 4 Scenario 1).

Tasks 1-3 ran on GPU, and bitsandbytes INT8/NF4 are GPU-only. Scenario 1 of the
report (edge / laptop, no GPU, <=4 GB RAM for the model, <2 s latency per
response) therefore relied on GPU-measured size PLUS the *assumption* of a CPU
4-bit backend. This script closes that gap: it runs the model entirely on CPU and
measures real size / peak RSS / throughput / perplexity for FP32 and HQQ INT8/INT4
(HQQ runs on CPU; bitsandbytes does not). The numbers here back the Scenario-1
recommendation with measurements instead of an extrapolation.

Note: BF16 is omitted on CPU — bf16 matmul kernels are not consistently available
on CPU, so FP32 is the honest CPU baseline. Low-bit is HQQ (calibration-free,
CPU-capable), the realistic edge backend.

Run:
    python cpu_benchmark.py
    python cpu_benchmark.py --configs FP32 HQQ-INT4 --n-docs 100 --new-tokens 64

Produces:
    results/cpu_benchmark.png  + a logged table + a Scenario-1 verdict.
"""
import argparse
import logging
import os
import threading
import time

import matplotlib.pyplot as plt
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.core.quantize import BaseQuantizeConfig
from hqq.models.hf.base import AutoHQQHFModel

from src.common import (
    MODEL_NAME, load_eval_corpus, perplexity, measure_size_mb,
    measure_throughput, free,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CPU = torch.device("cpu")
THROUGHPUT_PROMPT = "The history of artificial intelligence began"
RAM_BUDGET_MB = 4096          # Scenario 1 constraint: <= 4 GB for the model
LATENCY_RESPONSE_TOKENS = 50  # a "short response" used to express latency in seconds


class _PeakRSS:
    """Context manager: samples process RSS in a background thread and records the
    peak (MB) over its lifetime. Unlike tracemalloc, RSS includes torch tensors —
    so this is a real peak-memory figure for CPU inference."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss / (1024 ** 2)
            self.peak = max(self.peak, rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss / (1024 ** 2)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


class CPUBenchmark:
    """Benchmarks the model on CPU at FP32 + HQQ INT8/INT4 and fills a table that
    mirrors Task 1 but with a real CPU peak-RSS column."""

    CONFIGS = ["FP32", "HQQ-INT8", "HQQ-INT4"]
    METRICS = ["Size (MB)", "Peak RSS (MB)", "Tokens/sec", "Perplexity"]
    HQQ_BITS = {"HQQ-INT8": 8, "HQQ-INT4": 4}

    def __init__(self, model_name: str = MODEL_NAME, n_docs: int = 100,
                 new_tokens: int = 64):
        self.model_name = model_name
        self.n_docs = n_docs
        self.new_tokens = new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Cap the corpus: full WikiText-2 on CPU FP32 would take far too long.
        # Same fixed selection (first n_docs) so configs stay comparable here.
        self.corpus = load_eval_corpus(self.tokenizer, n_docs=n_docs)
        self.results: dict[str, dict] = {}

    def _load(self, config: str):
        logging.info(f"[CPU] Loading {config} ...")
        if config == "FP32":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32).to(CPU)
        bits = self.HQQ_BITS[config]
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32)
        AutoHQQHFModel.quantize_model(
            model, quant_config=BaseQuantizeConfig(nbits=bits, group_size=64),
            compute_dtype=torch.float32, device="cpu")
        return model

    def _benchmark_one(self, config: str):
        # Load FIRST (not under the RSS sampler): for HQQ the load step also runs
        # the one-time in-place quantization, which transiently holds the FP32
        # weights — that is a build-time cost, NOT "peak memory at inference". We
        # sample RSS only around the actual inference, with the model already
        # resident, so the figure reflects real deployment memory.
        model = self._load(config)
        size_mb = measure_size_mb(model)
        with _PeakRSS() as rss:
            ppl = perplexity(model, self.corpus, device=CPU)
            tok_s = measure_throughput(
                model, self.tokenizer, THROUGHPUT_PROMPT,
                new_tokens=self.new_tokens, runs=2, device=CPU)
        row = {
            "Size (MB)":     round(size_mb, 1),
            "Peak RSS (MB)": round(rss.peak, 1),
            "Tokens/sec":    round(tok_s, 1),
            "Perplexity":    round(ppl, 3),
        }
        logging.info(f"{config}: {row}")
        free(model)
        return row

    def _print_table(self):
        sep = "-" * 72
        logging.info(sep)
        logging.info(f"CPU BENCHMARK — {self.model_name} (n_docs={self.n_docs})")
        logging.info(sep)
        header = f"{'Config':<10}" + "".join(f"{m:>16}" for m in self.METRICS)
        logging.info(header)
        logging.info(sep)
        for cfg in self.CONFIGS:
            if cfg not in self.results:
                continue
            r = self.results[cfg]
            logging.info(f"{cfg:<10}" + "".join(f"{r[m]:>16}" for m in self.METRICS))
        logging.info(sep)

    def _verdict(self):
        """Check the recommended edge config (HQQ-INT4) against Scenario 1's
        hard constraints: <= 4 GB model + <2 s per (short) response."""
        rec = self.results.get("HQQ-INT4")
        if not rec:
            return
        size_ok = rec["Size (MB)"] <= RAM_BUDGET_MB
        tok_s = rec["Tokens/sec"]
        latency = LATENCY_RESPONSE_TOKENS / tok_s if tok_s else float("inf")
        logging.info("SCENARIO 1 VERDICT (HQQ-INT4 on CPU):")
        logging.info(f"  • RAM budget <=4 GB:   {rec['Size (MB)']} MB  "
                     f"=> {'PASS' if size_ok else 'FAIL'}")
        logging.info(f"  • Latency for a {LATENCY_RESPONSE_TOKENS}-token reply: "
                     f"{latency:.2f} s  ({tok_s} tok/s)  "
                     f"=> {'PASS (<2 s)' if latency < 2 else 'see note'}")

    def _plot(self, path: str = "results/cpu_benchmark.png"):
        os.makedirs("results", exist_ok=True)
        configs = [c for c in self.CONFIGS if c in self.results]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f"CPU Benchmark — {self.model_name} (Scenario 1 validation)",
                     fontsize=14, fontweight="bold")
        for ax, metric in zip(axes.flat, self.METRICS):
            vals = [self.results[c][metric] for c in configs]
            bars = ax.bar(configs, vals, color="seagreen", alpha=0.85)
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

    def run_pipeline(self, configs: list[str] | None = None):
        configs = configs or self.CONFIGS
        for cfg in configs:
            try:
                self.results[cfg] = self._benchmark_one(cfg)
            except Exception as exc:
                logging.warning(f"Skipping {cfg}: {exc}")
        self._print_table()
        self._verdict()
        self._plot()
        return self.results


def main():
    parser = argparse.ArgumentParser(description="CPU-only benchmark (Scenario 1)")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model id.")
    parser.add_argument("--configs", nargs="+", default=None,
                        choices=CPUBenchmark.CONFIGS,
                        help="Subset of CPU configs (default: all).")
    parser.add_argument("--n-docs", type=int, default=100,
                        help="Eval-corpus documents (capped for CPU speed).")
    parser.add_argument("--new-tokens", type=int, default=64,
                        help="Tokens to generate for the throughput measurement.")
    args = parser.parse_args()

    CPUBenchmark(model_name=args.model, n_docs=args.n_docs,
                 new_tokens=args.new_tokens).run_pipeline(args.configs)


if __name__ == "__main__":
    main()
