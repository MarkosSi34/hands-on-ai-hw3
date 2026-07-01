import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import torch
import bitsandbytes.functional as bnbF
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (
    MODEL_NAME, DEVICE, load_eval_corpus, perplexity, measure_throughput, free,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Calibration MUST come from the WikiText-2 TRAINING split — never the eval
# split (same data-leakage principle as Homework 1).
N_CALIB = 128
GPTQ_GROUP_SIZE = 128
GPTQ_DIR = "gptq_model"
THROUGHPUT_PROMPT = "The history of artificial intelligence began"

# The single attention matrix whose distribution we visualise (e.g. layer-0
# query projection). Adjust the substring to match your model's naming.
WEIGHT_KEY_SUBSTR = "layers.0.self_attn.q_proj"


class GPTQComparison:
    """
    Compares BF16 vs naive NF4 vs GPTQ INT4 on the fixed eval corpus, plots the
    weight distribution of one attention matrix, and reports the GPTQ
    perplexity advantage as an absolute and percentage reduction over naive INT4.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.corpus = load_eval_corpus(self.tokenizer)
        self.results: dict[str, dict] = {}
        self.calib_minutes = None

    # Calibration data (TRAINING split only)
    def _calibration_samples(self):
        from datasets import load_dataset
        logging.info(f"Building {N_CALIB} GPTQ calibration samples from the "
                     f"WikiText-2 TRAINING split...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        samples, i = [], 0
        for row in ds:
            text = row["text"].strip()
            if len(text) < 64:
                continue
            samples.append(text)  # gptqmodel tokenises internally
            i += 1
            if i >= N_CALIB:
                break
        return samples

    # Loading each config
    def _load_bf16(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16).to(DEVICE)

    def _load_nf4(self):
        from transformers import BitsAndBytesConfig
        return AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16))

    def _load_or_build_gptq(self):
        """Build the GPTQ model once, cache to disk, reload on later runs.

        Uses gptqmodel — the maintained successor to auto-gptq. auto-gptq is
        abandoned and no longer imports against transformers 5.x (it pulls the
        removed `no_init_weights` symbol), so we switched backends.
        """
        from gptqmodel import GPTQModel, QuantizeConfig, BACKEND

        # Inference kernel from env GPTQ_BACKEND (default TORCH). TORCH is the
        # pure-PyTorch path that runs on ANY GPU, incl. the GTX 1660 (Turing
        # sm_75, no CUDA toolkit) where the default Marlin kernel fails to load.
        # On an Ampere+ GPU (sm_80+) set GPTQ_BACKEND=auto for the fast Marlin path.
        backend = getattr(BACKEND, os.environ.get("GPTQ_BACKEND", "TORCH").upper(),
                          BACKEND.TORCH)

        if os.path.isdir(GPTQ_DIR):
            logging.info(f"Loading cached GPTQ model from '{GPTQ_DIR}' (backend={backend})...")
            return GPTQModel.load(GPTQ_DIR, backend=backend)

        logging.info("Calibrating GPTQ (this takes several minutes)...")
        quantize_config = QuantizeConfig(bits=4, group_size=GPTQ_GROUP_SIZE)
        model = GPTQModel.load(self.model_name, quantize_config)

        calib = self._calibration_samples()
        t0 = time.perf_counter()
        model.quantize(calib, tokenizer=self.tokenizer, batch_size=1)
        self.calib_minutes = (time.perf_counter() - t0) / 60.0
        logging.info(f"GPTQ calibration took {self.calib_minutes:.2f} min.")

        model.save(GPTQ_DIR)
        free(model)                           # reload from disk for clean inference kernels
        return GPTQModel.load(GPTQ_DIR, backend=backend)

    # Weight distribution plot
    def _get_attn_weights(self, model):
        """Return the flattened, DEQUANTIZED weights of the chosen matrix so all
        three panels share the same real weight-value x-axis.
        """
        for name, module in model.named_modules():
            if WEIGHT_KEY_SUBSTR not in name or not name.endswith("q_proj"):
                continue
            # GPTQ packed linear: reconstruct the float weights it encodes.
            if hasattr(module, "dequantize_weight"):
                try:
                    W = module.dequantize_weight()
                    return W.detach().float().flatten().cpu()
                except Exception as exc:
                    logging.warning(f"GPTQ dequantize failed on {name}: {exc}")
                    return None
            w = getattr(module, "weight", None)
            if w is None:
                continue
            # bitsandbytes NF4: the Params4bit carries a quant_state to invert.
            qs = getattr(w, "quant_state", None)
            if qs is not None:
                W = bnbF.dequantize_4bit(w.data, quant_state=qs)
                return W.detach().float().flatten().cpu()
            # plain float weight (BF16 baseline).
            return w.detach().float().flatten().cpu()
        logging.warning(f"No q_proj module matched '{WEIGHT_KEY_SUBSTR}'.")
        return None

    def _plot_weight_distributions(self, weights: dict[str, torch.Tensor],
                                   path: str = "results/task2_weight_distributions.png"):
        os.makedirs("results", exist_ok=True)
        # Filter NaN/inf (GPTQ dequant can emit them for some models) and drop any panel left empty
        present = []
        for label, w in weights.items():
            if w is None:
                continue
            w = w[torch.isfinite(w)]
            if w.numel() == 0:
                logging.warning(f"Panel '{label}' has no finite weights; skipping it.")
                continue
            present.append((label, w))
        if not present:
            logging.warning("No finite weights to plot; skipping weight-distribution plot.")
            return
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), sharey=True)
        if len(present) == 1:
            axes = [axes]
        fig.suptitle(f"Task 2 — Weight distribution of {WEIGHT_KEY_SUBSTR}",
                     fontsize=13, fontweight="bold")
        for ax, (label, w) in zip(axes, present):
            ax.hist(w.numpy(), bins=120, color="steelblue", alpha=0.85)
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("weight value")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("count")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logging.info(f"Saved '{path}'")

    def _plot_comparison(self, path: str = "results/task2_gptq_comparison.png"):
        os.makedirs("results", exist_ok=True)
        labels = [c for c in ("A_BF16", "B_NF4", "C_GPTQ") if c in self.results]
        ppls = [self.results[c]["ppl"] for c in labels]
        toks = [self.results[c].get("tok_s") for c in labels]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Task 2 — GPTQ vs Naive INT4", fontsize=13, fontweight="bold")

        b1 = ax1.bar(labels, ppls, color="tomato", alpha=0.85)
        for bar in b1:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        ax1.set_title("Perplexity (lower is better)")
        ax1.grid(True, axis="y", alpha=0.3)

        tok_labels = [l for l, t in zip(labels, toks) if t is not None]
        tok_vals = [t for t in toks if t is not None]
        b2 = ax2.bar(tok_labels, tok_vals, color="steelblue", alpha=0.85)
        for bar in b2:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
        ax2.set_title("Throughput (tokens/sec)")
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logging.info(f"Saved '{path}'")

    # Orchestration
    def run_pipeline(self):
        weights = {}

        # Config A — BF16 baseline
        model = self._load_bf16()
        self.results["A_BF16"] = {"ppl": perplexity(model, self.corpus)}
        weights["A: BF16"] = self._get_attn_weights(model)
        free(model)

        # Config B — naive NF4 INT4
        try:
            model = self._load_nf4()
            self.results["B_NF4"] = {
                "ppl": perplexity(model, self.corpus),
                "tok_s": measure_throughput(model, self.tokenizer, THROUGHPUT_PROMPT),
            }
            weights["B: Naive INT4"] = self._get_attn_weights(model)
            free(model)
        except Exception as exc:
            logging.warning(f"Skipping Config B (NF4): {exc}")

        # Config C — GPTQ INT4
        try:
            model = self._load_or_build_gptq()
            self.results["C_GPTQ"] = {
                "ppl": perplexity(model, self.corpus),
                "tok_s": measure_throughput(model, self.tokenizer, THROUGHPUT_PROMPT),
                "calib_min": self.calib_minutes,
            }
            weights["C: GPTQ INT4"] = self._get_attn_weights(model)
            free(model)
        except Exception as exc:
            logging.warning(f"Skipping Config C (GPTQ): {exc}")

        self._plot_weight_distributions(weights)
        self._plot_comparison()
        self._report()
        return self.results

    def _report(self):
        sep = "-" * 56
        logging.info(sep)
        logging.info("TASK 2 — GPTQ vs NAIVE INT4")
        logging.info(sep)
        for cfg, r in self.results.items():
            logging.info(f"{cfg:<8} ppl={r['ppl']:.3f} "
                         f"tok/s={r.get('tok_s', float('nan'))}")
        if "B_NF4" in self.results and "C_GPTQ" in self.results:
            b, c = self.results["B_NF4"]["ppl"], self.results["C_GPTQ"]["ppl"]
            abs_red = b - c
            pct_red = 100.0 * abs_red / b
            logging.info(sep)
            logging.info(f"GPTQ advantage over naive INT4: "
                         f"{abs_red:.3f} PPL ({pct_red:.1f}% reduction).")
        logging.info(sep)


def main():
    parser = argparse.ArgumentParser(description="Task 2: GPTQ vs Naive INT4")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    GPTQComparison(model_name=args.model).run_pipeline()


if __name__ == "__main__":
    main()
