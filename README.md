# Quantization in Practice | Hands-on AI Homework 3
**Φοιτητής:** Μάρκος Συρούκης  
**A.M.:** 09325023  
**Εξάμηνο:** 2ο  
**Μάθημα:** Hands-on AI Homework 3  
**Σχολή:** ΣΕΜΦΕ, ΕΜΠ  

---

Εφαρμογή των αλγορίθμων quantization (που υλοποιήθηκαν from-scratch στο lab) σε
ένα πραγματικό γλωσσικό μοντέλο, με τις βιβλιοθήκες παραγωγής **bitsandbytes**,
**AutoGPTQ** και **hqq**. Συγκρίνουμε επίπεδα ακρίβειας (FP32 → INT2),
αξιολογούμε το GPTQ έναντι naive rounding, μελετάμε την ευαισθησία στα δεδομένα
calibration, και καταλήγουμε σε τεκμηριωμένες προτάσεις deployment.

> **Σημείωση:** ο κώδικας του lab (NumPy) **δεν** χρησιμοποιείται εδώ· οι
> βιβλιοθήκες υλοποιούν τους ίδιους αλγορίθμους σε κλίμακα με hardware-optimised
> kernels.

---

## 1. Μοντέλο & Hardware

- **Μοντέλο:** `Qwen/Qwen2.5-0.5B` _(ίδιο σε όλα τα tasks — ορίζεται στο `src/common.py:MODEL_NAME`)_
- **Hardware:** NVIDIA GeForce **GTX 1660 (6 GB VRAM)**, CUDA 13 / torch 2.12, Python 3.11 _(Turing, χωρίς tensor cores)_
- **Evaluation corpus:** WikiText-2 test split, πρώτα 256 tokens κάθε εγγράφου,
  **σταθερό** σε όλα τα Tasks 1–3 (`src/common.load_eval_corpus`).

---

## 2. Δομή Project

```
.
├── pyproject.toml              # uv project + dependencies
├── requirements.txt            # human-readable direct deps
├── requirements.lock.txt       # ΑΚΡΙΒΕΙΣ pinned εκδόσεις (reproducible install)
├── Dockerfile / .dockerignore  # reproducible GPU image (CUDA devel base)
├── run_all.py                  # ΕΝΙΑΙΟ entry point: --tasks {1,2,3,bonus,ablation,all}
├── src/
│   └── common.py               # FIXED eval corpus, perplexity, μετρήσεις
├── task1_benchmark.py          # Task 1 — precision sweep (FP32/BF16/INT8/INT4)
├── task2_gptq.py               # Task 2 — GPTQ vs naive INT4
├── task3_calibration.py        # Task 3 — calibration domain sensitivity
├── bonus_2bit_hqq.py           # Bonus A — 2-bit HQQ + precision curve
├── ablation_gptq.py            # Extra — GPTQ group_size / n_calib ablation
├── report.md                   # Task 4 — deployment recommendations (≥500 λέξεις)
├── collaboration_report.md     # Bonus B — joint report (προαιρετικό)
└── results/                    # όλα τα PNG deliverables
```

> Τα quantized weights **δεν** commit-άρονται (μεγάλα + αναπαραγώγιμα — βλ.
> `.gitignore`). Τα GPTQ μοντέλα cache-άρονται τοπικά για επαναχρησιμοποίηση.

---

## 3. Εγκατάσταση (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .          # ή: uv pip install -r requirements.txt
```

---

## 4. Εκτέλεση — μία εντολή ανά task

```bash
python task1_benchmark.py      # → results/task1_benchmarks.png + πίνακας
python task2_gptq.py           # → results/task2_weight_distributions.png, task2_gptq_comparison.png
python task3_calibration.py    # → results/task3_calibration_study.png
python bonus_2bit_hqq.py       # → results/bonus_2bit_precision_curve.png (Bonus A)
```

…ή όλα μαζί με ένα entry point:

```bash
python run_all.py --tasks all                       # 1,2,3,bonus,ablation
python run_all.py --tasks 1 2 3                      # υποσύνολο
GPTQ_BACKEND=auto python run_all.py --tasks 2 3      # γρήγορο Marlin kernel σε Ampere+
python run_all.py --tasks 1 2 --model Qwen/Qwen2.5-1.5B   # scaling extension
```

> **`GPTQ_BACKEND`**: default `TORCH` (pure-PyTorch, τρέχει σε κάθε GPU incl. GTX 1660).
> Σε Ampere+ (sm_80+) βάλε `auto` για το βελτιστοποιημένο Marlin GPTQ kernel.

---

## 4¾. GPU session & αναπαραγωγιμότητα

Δύο σημεία απαιτούν GPU με **tensor cores** (που η GTX 1660 δεν έχει) — τα αφήνουμε
ως ρητά documented κενά (βλ. §4½) και τα γεμίζει ένα Ampere+ box:

- **INT8 (Task 1)** — bitsandbytes `LLM.int8()` τρέχει μόνο εκεί.
- **Πραγματικό GPTQ throughput** — με `GPTQ_BACKEND=auto` το Marlin kernel δίνει
  τα σωστά (χαμηλή ακρίβεια = ταχύτερη), σε αντίθεση με το τοπικό Torch fallback.

**Reproducible image** (μοντέλα κατεβαίνουν στο runtime, δεν ψήνονται στο image):

```bash
docker build -t hw3-quant .
docker run --gpus all -e GPTQ_BACKEND=auto \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/results:/app/results \
  hw3-quant --tasks all
```

Το CUDA *devel* base φέρνει `nvcc`, οπότε το Marlin kernel κάνει JIT-compile
(λύνει το `CUDA_HOME` πρόβλημα του dev box). Καρφωμένες εκδόσεις: `requirements.lock.txt`.

---

## 4½. Σημειώσεις περιβάλλοντος & αλλαγές υλοποίησης

> Καταγράφονται οι αποκλίσεις από το αρχικό scaffold που χρειάστηκαν για να
> τρέξει ο κώδικας στο πραγματικό περιβάλλον (Python 3.11, `transformers` 5.x,
> `torch` 2.12 + CUDA 13, GPU **NVIDIA GTX 1660 / 6 GB**).

1. **`wikitext` → `Salesforce/wikitext`** (`src/common.py`, `task2_gptq.py`).
   Το παλιό `wikitext` repo-id φορτώνει legacy *loading script* (`wikitext.py`)
   που η νέα `datasets` δεν υποστηρίζει (σφάλμα `Invalid HF URI`). Το canonical
   parquet mirror `Salesforce/wikitext` φορτώνει χωρίς script. Ίδια δεδομένα.

1b. **Task 3 datasets → parquet ισοδύναμα** (`task3_calibration.py`). Τα
   `codeparrot/github-code` (Python) και `pg19` (academic) είναι script-based
   (`RuntimeError: Dataset scripts are no longer supported`). Αντικαταστάθηκαν:
   Python → `google-research-datasets/mbpp` (eval = **test** split, calibration
   C2 = **train** split, ώστε να μην επικαλύπτονται)· academic C3 →
   `CShorten/ML-ArXiv-Papers` (abstracts). Το `wikimedia/wikipedia` (C1) είναι
   ήδη parquet — έμεινε ως έχει.

2. **`auto-gptq` → `gptqmodel`** (`task2_gptq.py`, `task3_calibration.py`,
   `pyproject.toml`, `requirements.txt`). Το `auto-gptq` είναι εγκαταλελειμμένο
   και **δεν κάνει import με `transformers` 5.x** (ψάχνει το αφαιρεμένο σύμβολο
   `no_init_weights`). Το `gptqmodel` είναι ο συντηρούμενος διάδοχος με
   ισοδύναμο GPTQ· νέο API: `GPTQModel.load(...)` / `model.quantize(calib,
   tokenizer=...)` / `model.save(...)`. Η calibration περνά πλέον ως λίστα
   **strings** (το tokenization γίνεται εσωτερικά). Παρ-εξάρτηση: `torchvision`
   (το `gptqmodel` το κάνει import eagerly κατά τη φόρτωση).
   - **Inference kernel = `BACKEND.TORCH`.** Το default kernel (Marlin) απαιτεί
     GPU Ampere (sm_80+) + JIT compilation με `CUDA_HOME`/`nvcc`. Η GTX 1660
     είναι Turing (sm_75) χωρίς CUDA toolkit → Marlin/ExLlama αποτυγχάνουν στο
     load. Το `BACKEND.TORCH` (pure-PyTorch) τρέχει παντού χωρίς compilation.
     Η ίδια η quantization (calibration) πέτυχε ανεξάρτητα — μόνο το inference
     kernel χρειαζόταν επιλογή.

3. **INT8 (bitsandbytes) δεν τρέχει στην GTX 1660.** Το `LLM.int8()` απαιτεί
   `cublasLt` INT8 matmul που η σειρά GTX 16xx (Turing, **χωρίς tensor cores**)
   δεν υποστηρίζει → `CUBLAS_STATUS_NOT_SUPPORTED` (status 15). Το Task 1
   κάνει gracefully skip τη γραμμή INT8· καλύπτουμε το trade-off με FP32/BF16/INT4.

4. **HQQ μέσω native API, όχι `HqqConfig`** (`bonus_2bit_hqq.py`). Το loading
   path του `HqqConfig` στο `transformers` 5.x δεν είναι ακόμη υλοποιημένο
   (`NotImplementedError: QuantizationMethod.HQQ is not available yet`). Φορτώνουμε
   FP16 μοντέλο και το κβαντίζουμε in-place με το native API του HQQ
   (`AutoHQQHFModel.quantize_model`).

---

## 5. Task 1 — Precision Benchmark _(αποτελέσματα)_

| Precision | Size (MB) | Memory (MB) | Tokens/sec | Perplexity |
|-----------|-----------|-------------|------------|------------|
| FP32      | 1884.6    | 2351.4      | 37.1       | 22.733     |
| BF16      | 942.3     | 1406.0      | 33.5       | 22.657     |
| INT8      | — _(skip)_ | —          | —          | — _(GTX 16xx: no cublasLt INT8 — βλ. §4½)_ |
| INT4      | 430.4     | 911.6       | 24.1       | 27.504     |

_Σχόλιο: σε ποιο επίπεδο ακρίβειας το trade-off ποιότητας–αποδοτικότητας είναι
πιο ευνοϊκό; Σύνδεση με τα σφάλματα quantization του lab._

---

## 6. Task 2 — GPTQ vs Naive INT4 _(αποτελέσματα)_

| Config | Method | Perplexity | Tokens/sec |
|--------|--------|------------|------------|
| A | BF16 baseline | 22.657 | — |
| B | Naive INT4 (NF4) | 27.504 | 23.1 |
| C | GPTQ INT4 | **26.062** | 11.4 † |

- **GPTQ advantage έναντι naive INT4:** **1.442** PPL absolute, **5.2 %** reduction
  (GPTQ ανακτά ~30 % της απώλειας ποιότητας BF16→INT4).
- **GPTQ calibration time:** **2.63** λεπτά (128 samples, WikiText-2 **train** split).
- † _Το GPTQ tok/s **δεν** είναι αντιπροσωπευτικό: τρέχει με `BACKEND.TORCH`
  (unoptimized) — το `torch.compile` κάνει skip το bf16 στη GTX 1660 (sm_75).
  Το optimized Marlin kernel απαιτεί Ampere (sm_80+). Σύγκριναμε **PPL**, όχι ταχύτητα._
- _Η κατανομή βαρών για το GPTQ (panel C) λείπει: το `gptqmodel` αποθηκεύει
  packed `qweight` (όχι float `q_proj.weight`), οπότε το ιστόγραμμα δείχνει A & B._

---

## 7. Task 3 — Calibration Data Sensitivity _(αποτελέσματα)_

_Eval corpora: WikiText-2 test (English) & MBPP test (Python). Calibration:
C1 `wikimedia/wikipedia`, C2 `mbpp` train, C3 `ML-ArXiv-Papers` (βλ. §4½)._

| Config | Calibration domain | WikiText-2 PPL | Python code PPL |
|--------|--------------------|----------------|-----------------|
| BF16 baseline | — | 22.657 | 3.378 |
| GPTQ C1 | Wikipedia | **27.028** ✅ | 4.699 |
| GPTQ C2 | Python | 60.284 | **3.582** ✅ |
| GPTQ C3 | Academic | 28.094 | 4.272 |

✅ = καλύτερο GPTQ variant στη στήλη του (εξαιρώντας το BF16 baseline).

**Συμπέρασμα — το domain match έχει σημασία, και μάλιστα δραματικά:**
- Η **διαγώνιος νικάει**: C1 (Wikipedia) δίνει το χαμηλότερο WikiText PPL,
  C2 (Python) δίνει το χαμηλότερο Python PPL. Κάθε calibration domain παράγει
  το καλύτερο μοντέλο **στο δικό του domain**.
- **Cross-domain penalty:** το C2 (code-calibrated) εκτοξεύεται σε **60.28** PPL
  στα Αγγλικά — **2.6×** χειρότερα από το BF16 (22.66). Η calibration μόνο σε
  κώδικα καταστρέφει το γλωσσικό μοντέλο για prose.
- _Οι τιμές Python PPL (3.4–4.7) είναι χαμηλές γιατί τα MBPP solutions είναι
  σύντομα/προβλέψιμα· σύγκριση μόνο **εντός** στήλης, όχι μεταξύ στηλών._

---

## 7½. Bonus A — 2-bit (HQQ) & quality floor _(αποτελέσματα)_

Πλήρης precision curve FP32 → BF16 → INT4 → INT2 (`results/bonus_2bit_precision_curve.png`).

| Precision | Size (MB) | Perplexity |
|-----------|-----------|------------|
| FP32 | 1884.6 | 22.733 |
| BF16 | 942.3 | 22.657 |
| INT4 (NF4) | 430.4 | 27.504 |
| **INT2 (HQQ)** | **345.1** | **312339** ⚠️ |

**Συμπέρασμα — το quality floor είναι στα 2-bit:** το INT2 **καταρρέει** (PPL
312k έναντι 27.5 στα INT4). Το HQQ εδώ είναι calibration-free RTN — χωρίς
Hessian error-compensation (όπως το GPTQ) ή QAT, τα 4 επίπεδα ανά βάρος δεν
αρκούν για ένα μικρό μοντέλο 0.5B. _Σημείωση μεγέθους:_ το INT2 δεν είναι 4×
μικρότερο από το INT4 γιατί ο tied embedding/`lm_head` πίνακας (151936×896 ≈
27 % του μοντέλου) μένει FP16 (~272 MB) — το HQQ κβαντίζει μόνο `Linear` layers.
_INT8 παραλείπεται (GTX 16xx — βλ. §4½)._

---

## 8. Παραδοτέα (checklist)

- [x] `task1_benchmark.py` — πίνακας + `results/task1_benchmarks.png`
- [x] `task2_gptq.py` — `results/task2_weight_distributions.png`, `results/task2_gptq_comparison.png`
- [x] `task3_calibration.py` — `results/task3_calibration_study.png`
- [x] `report.md` — Task 4 (≥500 λέξεις)
- [x] `requirements.txt` + `pyproject.toml`
- [x] `README.md` (αποτελέσματα Tasks 1–3 + Bonus A συμπληρωμένα)
- [x] _(Bonus A)_ `results/bonus_2bit_precision_curve.png`
- [ ] _(Bonus B)_ `collaboration_report.md`
