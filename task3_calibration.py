"""
Task 3 — Calibration Data Sensitivity.

Run GPTQ three times with IDENTICAL hyperparameters and model but DIFFERENT
calibration corpora, then cross-evaluate each variant on two domains.

Calibration corpora (128 samples each)
    C1  General Wikipedia   wikimedia/wikipedia (20231101.en)
    C2  Python code         google-research-datasets/mbpp (train split)
    C3  Academic / formal   CShorten/ML-ArXiv-Papers (abstracts)

Evaluation corpora
    WikiText-2 test   — general English prose (the fixed Task 1–2 corpus)
    Python code       — 50 short Python functions from MBPP (test split)

NOTE: the original codeparrot/github-code and pg19 sources are legacy
loading-script datasets, no longer supported by `datasets`; the parquet
datasets above replace them (see README §4½).

Run:
    python task3_calibration.py

Produces:
    results/task3_calibration_study.png   — grouped bars, both eval domains
    cached models under gptq_C1/ C2/ C3/  (NOT committed — see .gitignore)
"""
import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import MODEL_NAME, DEVICE, load_eval_corpus, perplexity, free

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

N_CALIB = 128
GPTQ_GROUP_SIZE = 128
N_PYTHON_EVAL = 50
# Truncate each calibration doc to this many characters (~500 tokens). Full
# Wikipedia articles (C1) are thousands of tokens and OOM the 6 GB GPU during
# calibration; bounding length also keeps the three domains comparable.
CALIB_MAX_CHARS = 2000

# (run_id, human label, cache dir). Calibration loaders are below.
RUNS = [
    ("C1", "Wikipedia", "gptq_C1"),
    ("C2", "Python",    "gptq_C2"),
    ("C3", "Academic",  "gptq_C3"),
]


class CalibrationStudy:
    """
    Quantifies whether the DOMAIN of GPTQ calibration data matters by holding
    everything else fixed and varying only the 128-sample calibration set.

    Fills the Task 3 table: BF16 baseline + three GPTQ variants, each scored on
    both WikiText-2 and Python-code perplexity.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eval_wiki = load_eval_corpus(self.tokenizer)       # general English
        self.eval_python = self._load_python_eval()             # code
        self.results: dict[str, dict] = {}

    # Evaluation corpora
    def _load_python_eval(self):
        """50 short Python functions from the MBPP TEST split (parquet).

        MBPP replaces codeparrot/github-code, which is a legacy loading-script
        dataset no longer supported by datasets. Using the *test* split keeps
        the eval set disjoint from the *train* split used for C2 calibration.
        """
        from datasets import load_dataset
        logging.info("Loading Python-code evaluation corpus (MBPP test)...")
        ds = load_dataset("google-research-datasets/mbpp", split="test",
                          streaming=True)
        samples = []
        for row in ds:
            code = row["code"]
            if "def " not in code or len(code) < 64:
                continue
            ids = self.tokenizer(code, return_tensors="pt").input_ids[0][:256]
            if ids.numel() >= 2:
                samples.append(ids)
            if len(samples) >= N_PYTHON_EVAL:
                break
        logging.info(f"Python eval corpus: {len(samples)} functions.")
        return samples

    # Calibration loaders (TRAINING-domain data only)
    def _calib_wikipedia(self):
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True)
        return self._take_calib(ds, key="text")

    def _calib_python(self):
        from datasets import load_dataset
        # MBPP train split (parquet) — replaces script-based codeparrot/github-code.
        ds = load_dataset("google-research-datasets/mbpp", split="train",
                          streaming=True)
        return self._take_calib(ds, key="code")

    def _calib_academic(self):
        from datasets import load_dataset
        # ML-ArXiv-Papers abstracts (parquet) — formal/academic domain; replaces
        # the script-based pg19 long-form-text dataset.
        ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train",
                          streaming=True)
        return self._take_calib(ds, key="abstract")

    def _take_calib(self, ds, key: str):
        samples = []
        for row in ds:
            text = row[key].strip()
            if len(text) < 64:
                continue
            samples.append(text[:CALIB_MAX_CHARS])   # truncate; gptqmodel tokenises internally
            if len(samples) >= N_CALIB:
                break
        return samples

    _CALIB_FN = {
        "C1": "_calib_wikipedia",
        "C2": "_calib_python",
        "C3": "_calib_academic",
    }

    # GPTQ build / cache
    def _load_or_build(self, run_id: str, cache_dir: str):
        # gptqmodel replaces the abandoned auto-gptq (see task2_gptq.py).
        # Inference kernel from env GPTQ_BACKEND (default TORCH — pure-PyTorch,
        # runs on the GTX 1660 sm_75; set GPTQ_BACKEND=auto on Ampere+ for Marlin).
        from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
        backend = getattr(BACKEND, os.environ.get("GPTQ_BACKEND", "TORCH").upper(),
                          BACKEND.TORCH)
        if os.path.isdir(cache_dir):
            logging.info(f"Loading cached {run_id} from '{cache_dir}' (backend={backend})...")
            return GPTQModel.load(cache_dir, backend=backend)

        logging.info(f"Calibrating {run_id} (domain-specific GPTQ)...")
        cfg = QuantizeConfig(bits=4, group_size=GPTQ_GROUP_SIZE)
        model = GPTQModel.load(self.model_name, cfg)
        calib = getattr(self, self._CALIB_FN[run_id])()
        model.quantize(calib, tokenizer=self.tokenizer, batch_size=1)
        model.save(cache_dir)
        free(model)
        return GPTQModel.load(cache_dir, backend=backend)

    # ── Output 
    def _plot(self, path: str = "results/task3_calibration_study.png"):
        os.makedirs("results", exist_ok=True)
        labels = list(self.results.keys())
        wiki = [self.results[l]["wiki"] for l in labels]
        py = [self.results[l]["python"] for l in labels]

        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(10, 6))
        b1 = ax.bar(x - width / 2, wiki, width, label="WikiText-2 PPL",
                    color="steelblue", alpha=0.85)
        b2 = ax.bar(x + width / 2, py, width, label="Python-code PPL",
                    color="tomato", alpha=0.85)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title("Task 3 — Calibration Domain Sensitivity",
                     fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logging.info(f"Saved '{path}'")

    def _print_table(self):
        sep = "-" * 64
        logging.info(sep)
        logging.info(f"{'Config':<18}{'WikiText-2 PPL':>18}{'Python code PPL':>20}")
        logging.info(sep)
        for label, r in self.results.items():
            logging.info(f"{label:<18}{r['wiki']:>18.3f}{r['python']:>20.3f}")
        logging.info(sep)

    # ── Orchestration ─────────────────────────────────────────────────────
    def run_pipeline(self):
        # BF16 baseline (no calibration)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16).to(DEVICE)
        self.results["BF16 baseline"] = {
            "wiki": perplexity(model, self.eval_wiki),
            "python": perplexity(model, self.eval_python),
        }
        free(model)

        # Three domain-calibrated GPTQ variants
        for run_id, label, cache_dir in RUNS:
            try:
                model = self._load_or_build(run_id, cache_dir)
                self.results[f"GPTQ {run_id} ({label})"] = {
                    "wiki": perplexity(model, self.eval_wiki),
                    "python": perplexity(model, self.eval_python),
                }
                free(model)
            except Exception as exc:
                logging.warning(f"Skipping {run_id}: {exc}")

        self._print_table()
        self._plot()
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Task 3 — Calibration Data Sensitivity")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.parse_args()
    CalibrationStudy().run_pipeline()


if __name__ == "__main__":
    main()
