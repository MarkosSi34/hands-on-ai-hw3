"""
Shared utilities for Homework 3 — Quantization.

Everything that MUST be identical across Tasks 1–3 lives here so the
size–speed–quality numbers stay comparable between configurations:

  * MODEL_NAME            — the single model used throughout all four tasks.
  * load_eval_corpus()    — the FIXED WikiText-2 evaluation corpus
                            (first 256 tokens of each document). Fix this
                            once, reuse everywhere — if tokenisation or
                            sample selection drifts between runs, the
                            perplexity comparisons become meaningless.
  * perplexity()          — PPL = exp(mean cross-entropy) over a corpus.
  * measure_*()           — size on disk, peak inference memory, throughput.

The assignment forbids re-using the from-scratch NumPy lab code here; this
module only wires together HuggingFace + bitsandbytes / AutoGPTQ / hqq.
"""
import gc
import logging
import time
import tracemalloc

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── The single model used across ALL four tasks ───────────────────────────
# Pick ONE based on your hardware (see the assignment's Model Selection table)
# and DO NOT change it between tasks.
#   HuggingFaceTB/SmolLM2-360M   — CPU only
#   Qwen/Qwen2.5-0.5B            — CPU only (recommended)
#   Qwen/Qwen2.5-1.5B           — GPU >= 6 GB VRAM
#   meta-llama/Llama-3.2-1B     — GPU >= 4 GB VRAM
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# ── Fixed evaluation-corpus parameters ────────────────────────────────────
EVAL_TOKENS_PER_DOC = 256          # first N tokens of each WikiText-2 doc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_corpus(tokenizer, n_docs: int | None = None,
                     tokens_per_doc: int = EVAL_TOKENS_PER_DOC):
    """
    Build the FIXED WikiText-2 test evaluation corpus.

    Returns a list of 1-D LongTensors, one per document, each truncated to
    the first `tokens_per_doc` tokens. Call this once at the start of every
    task and feed the SAME list to every configuration.

    Parameters
    ----------
    tokenizer       : the model tokenizer (for consistent tokenisation).
    n_docs          : cap on number of documents (None = all non-empty docs).
    tokens_per_doc  : truncate each document to this many tokens.
    """
    from datasets import load_dataset

    logging.info("Loading WikiText-2 test split for the fixed eval corpus...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    samples = []
    for row in ds:
        text = row["text"].strip()
        if not text:                       # skip blank / heading-only lines
            continue
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.numel() < 2:                # need at least 2 tokens for a loss
            continue
        samples.append(ids[:tokens_per_doc])
        if n_docs is not None and len(samples) >= n_docs:
            break

    logging.info(f"Fixed eval corpus: {len(samples)} documents "
                 f"(<= {tokens_per_doc} tokens each).")
    return samples


@torch.no_grad()
def perplexity(model, corpus, device=DEVICE) -> float:
    """
    PPL = exp(mean cross-entropy) over the fixed corpus.

    The loss is averaged across ALL samples BEFORE exponentiating, exactly
    as the assignment specifies. Token-weighted so longer documents count
    proportionally.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for tokens in corpus:
        tokens = tokens.unsqueeze(0).to(device)
        out = model(input_ids=tokens, labels=tokens)
        # out.loss is the MEAN CE over (len-1) predicted tokens for this doc.
        n = tokens.numel() - 1
        total_loss += out.loss.item() * n
        total_tokens += n

    mean_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(mean_loss)))


def measure_size_mb(model) -> float:
    """Model size in MB via get_memory_footprint() (counts quantized bytes)."""
    return model.get_memory_footprint() / (1024 ** 2)


@torch.no_grad()
def measure_throughput(model, tokenizer, prompt: str, new_tokens: int = 200,
                       runs: int = 3, device=DEVICE) -> float:
    """
    Inference throughput (tokens/sec): generate exactly `new_tokens` from a
    fixed prompt, average over `runs`, EXCLUDING the first (warm-up) run.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    rates = []
    for i in range(runs + 1):              # +1 warm-up run that we discard
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i > 0:                          # i == 0 is warm-up
            rates.append(new_tokens / dt)
    return sum(rates) / len(rates)


def measure_peak_memory_cpu(fn):
    """
    Peak memory (MB) of calling `fn()` on CPU via tracemalloc.
    On GPU use torch.cuda.max_memory_allocated() around the call instead
    (see the reset_peak_memory / read_peak_memory helpers below).
    """
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 ** 2)


def reset_peak_memory(device=DEVICE):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def read_peak_memory_mb(device=DEVICE) -> float | None:
    """Peak GPU memory since last reset; None on CPU (use tracemalloc there)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def free(model=None):
    """Release a model and empty the CUDA cache between configurations."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
