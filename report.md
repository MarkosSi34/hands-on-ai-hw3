# Task 4: Αναφορά Προτάσεων Deployment

Όλα τα νούμερα παρακάτω προέρχονται από τις δικές μας μετρήσεις στο
`Qwen/Qwen2.5-0.5B`. Τα τελικά benchmarks έτρεξαν σε **rented RTX 4090** (Ada
sm_89, vast.ai), αφού η dev GTX 1660 (Turing sm_75, χωρίς tensor cores) δεν
υποστήριζε INT8 ούτε το Marlin GPTQ kernel (βλ. README §2). Baselines:
perplexity FP32 **22.733**, BF16 **22.664**. Στην RTX 4090 το throughput είναι
αντιπροσωπευτικό (χαμηλότερη ακρίβεια → ταχύτερα, με tensor cores), οπότε
χρησιμοποιούμε **μέγεθος, perplexity ΚΑΙ ταχύτητα** ως αξιόπιστες ενδείξεις.

---

## Σενάριο 1: Edge / CPU deployment

*Ελαφρύς τοπικός βοηθός σε laptop, χωρίς dedicated GPU. Περιορισμοί:
≤4 GB RAM για το μοντέλο, latency <2 s ανά απάντηση.*

**Προτεινόμενη ακρίβεια + μέθοδος:** **INT4 (4-bit, group-wise) με *βελτιστοποιημένο*
CPU kernel (GGUF / llama.cpp)** για τη μνήμη. **Προσοχή στο latency:** το naive HQQ
PyTorch path *δεν* αρκεί (δες μετρήσεις CPU πιο κάτω)· το GPU-only bitsandbytes ούτε
τρέχει σε CPU.

**Τεκμηρίωση (στοιχεία από Tasks):**
- **Μέγεθος: 430.4 MB** (Task 1, γραμμή INT4), μείωση **4.4×** από τα 1884.6 MB του
  FP32. Χωράει άνετα στο budget των 4 GB και σημαίνει **4.4× λιγότερη μνήμη βαρών προς
  streaming ανά token**, το κυρίαρχο κόστος του CPU inference.
- **Μνήμη: 911.6 MB** peak (Task 1) έναντι 2351.4 MB για FP32, το μοντέλο μαζί με
  τα activations χωρά άνετα σε ένα περιορισμένο laptop.

**Επικύρωση σε πραγματικό CPU (`cpu_benchmark.py`).** Επειδή τα Tasks 1–3 έτρεξαν σε
GPU (και το bitsandbytes είναι GPU-only), μετρήσαμε το edge σενάριο **απευθείας σε
CPU** (GTX-1660 box, GPU κρυμμένο με `CUDA_VISIBLE_DEVICES=""`), με HQQ (calibration-
free, CPU-capable). WikiText-2 (50 docs), βλ. README §9:

| Config (CPU) | Size (MB) | Peak RSS (MB) | Tokens/sec | Perplexity |
|---|---|---|---|---|
| FP32 | 1884.6 | 3286 | **6.6** | 24.177 |
| HQQ-INT8 | 860.7 | 3881 | 1.7 | 24.223 |
| HQQ-INT4 | 690.1 | 4268 | 1.4 | 29.236 |

Δύο εμπειρικά ευρήματα που **διορθώνουν** τη naive υπόθεση «INT4 = καλύτερο για edge»:
1. **Στον δίσκο το quantization κερδίζει:** HQQ-INT4 690 MB, **2.7× μικρότερο** από
   FP32 και ο περιορισμός «≤4 GB για το μοντέλο» ικανοποιείται άνετα.
2. **Στο latency, όμως, το HQQ σε CPU χάνει δραματικά:** **1.4 tok/s (INT4) vs 6.6
   tok/s (FP32)**  το 4-bit είναι **~4.7× πιο αργό**, γιατί το HQQ κάνει dequantization
   on-the-fly χωρίς optimized CPU kernel (γι' αυτό και το peak RSS *ανεβαίνει* στο
   inference αντί να πέφτει). Καμία ρύθμιση δεν πιάνει το <2 s με το PyTorch backend.

**Trade-off που δεχόμαστε:** (α) Ποιότητα: η perplexity ανεβαίνει **~21 %** στο
INT4 (22.664→27.502 GPU· 24.18→29.24 στο CPU subset), ανεκτό για καθημερινό βοηθό. (β)
**Το κρίσιμο μάθημα από τα CPU δεδομένα:** το κέρδος μνήμης του 4-bit είναι πραγματικό,
αλλά η *ταχύτητα* απαιτεί τον σωστό kernel, το naive HQQ δείχνει το concept (μέγεθος)
αλλά όχι το latency· ένα **GGUF/llama.cpp INT4** με πραγματικά quantized CPU kernels
είναι η μόνη ρεαλιστική διαδρομή για <2 s. Το **INT4 είναι και το πρακτικό κατώφλι
ποιότητας**: το Bonus A δείχνει ότι τα 2-bit (HQQ, calibration-free) **καταρρέουν**
(PPL **312k**), άρα δεν έχει νόημα να πάμε χαμηλότερα.

**Αξίζει το domain-matched calibration;** **Όχι.** Οι προτεινόμενες μέθοδοι 4-bit για
CPU (HQQ / NF4) είναι *calibration-free* round-to-nearest σχήματα, οπότε το ερώτημα
του Task 3 δεν εφαρμόζεται. Αν χρησιμοποιούσαμε GPTQ, ένας γενικός βοηθός θα έπρεπε να
καλιμπράρεται σε **ποικίλο γενικό κείμενο** (το καθεστώς C1/Wikipedia), ποτέ σε ένα
στενό domain, το Task 3 δείχνει ότι το στενό calibration είναι ενεργά επιβλαβές
(βλ. Σενάριο 3).

---

## Σενάριο 2: Server inference, βελτιστοποιημένο για throughput

*Single GPU server (π.χ. A100 40 GB), εκατοντάδες requests/min. Περιορισμός:
μεγιστοποίηση tokens/sec· υποβάθμιση ποιότητας <5 % σχετική αύξηση perplexity.*

**Προτεινόμενη ακρίβεια + μέθοδος:** **BF16** ως ασφαλές default (ταχύτητα +
ποιότητα). Αν χρειάζεται συμπίεση εντός του 5 %, **INT8**· το 4-bit μόνο αν το
budget χαλαρώσει.

**Τεκμηρίωση (στοιχεία απο Tasks):**
- **Σχετική PPL έναντι BF16 (=22.664):** **BF16 −0.3 %** vs FP32 (lossless) **και
  το ταχύτερο: 58.9 tok/s > FP32 52.7** (tensor cores). **INT8 22.936 = +1.2 %**
  είναι *εντός* του budget 5 %. Αντίθετα **GPTQ INT4 26.215 (+15.7 %)** και **NF4
  27.502 (+21.3 %)** *ξεπερνούν* το budget στο 0.5 B.
- **Ταχύτητα/μνήμη:** BF16 942 MB @ 58.9 tok/s. Το **INT8 ικανοποιεί την ποιότητα
  αλλά είναι αργό (10.4 tok/s)**, το bitsandbytes `LLM.int8()` κάνει mixed-precision
  outlier decomposition (για μνήμη, όχι throughput). Άρα για *throughput*-server το
  BF16 παραμένει η σωστή επιλογή· το INT8 ταιριάζει όταν η μνήμη (όχι το latency)
  είναι ο περιορισμός.

**Trade-off που δεχόμαστε:** για τον αυστηρό φάκελο 5 % κρατάμε BF16/INT8 και
παραιτούμαστε από τη 2× συμπίεση του 4-bit. Αν ο φάκελος χαλάρωνε, το **GPTQ INT4
είναι η σωστή επιλογή 4-bit** καθώς ανακτά **1.286 PPL (4.7 %)** μέσω Hessian
compensation **και τρέχει ταχύτερα από το NF4 (46.9 vs 35.6 tok/s, Marlin)**. Το
scaling extension το επιβεβαιώνει: στο **1.5 B η ποινή του INT4 πέφτει σε +13.7 %**
(από +21 % στο 0.5 B) και το INT8 μένει **+0.9 %** δηλαδή το budget 5 % γίνεται
ευκολότερο σε κλίμακα ενώ το 0.5 B είναι worst case μικρού μοντέλου.

**Αξίζει το domain-matched calibration;** **Όχι για γενική κίνηση server.** Με ευρύ
μείγμα requests, το γενικού σκοπού calibration είναι το σωστό. Αξίζει μόνο αν η κίνηση
είναι μετρήσιμα στραμμένη προς ένα domain αλλά ακόμα και τότε είναι το ίδιο επιχείρημα με το
Σενάριο 3.

---

## Σενάριο 3: Domain-specific (code) deployment

*Code assistant σε ιδιόκτητα Python codebases, single consumer GPU
(RTX 3080, 10 GB VRAM). Περιορισμός: διατήρηση καλής code perplexity μετά το
quantization.*

**Προτεινόμενη ακρίβεια + μέθοδος:** **GPTQ INT4 καλιμπραρισμένο σε Python code
(config C2).**

**Τεκμηρίωση (στοιχεία απο Tasks, με έμφαση στο Task 3):**
- **Python-code PPL με code-matched calibration (C2): 3.573** έναντι **Wikipedia
  calibration (C1): 4.634** δηλαδή **~23 % χαμηλότερη** code perplexity καθαρά από το
  ταίριασμα του domain calibration με το workload, με **μηδενικό επιπλέον κόστος**
  (ίδια bits, ίδιο group size, ίδια 128 samples). Το C2 νικά και το academic
  calibration (C3, 4.166).
- **VRAM footprint: 430.4 MB** (Task 1 INT4) χωρά στην RTX 3080 των 10 GB ~20× φορές,
  αφήνοντας άφθονο χώρο για KV-cache και long-context code prompts. Bonus: η RTX 3080
  είναι **Ampere (sm_86)**, άρα τρέχει το optimised Marlin GPTQ kernel και θα ήταν εδώ
  *γρήγορη* σε αντίθεση με την GTX 1660, που έπεσε στο αργό Torch kernel.

**Trade-off που δεχόμαστε:** το C2 είναι **καταστροφικό στα Αγγλικά (WikiText PPL
61.816, 2.7× χειρότερα από το baseline)**. Αυτό είναι αποδεκτό *επειδή πρόκειται για
ένα αποκλειστικό code model*, όχι γενικό βοηθό ουσιαστικά το ειδικεύουμε εσκεμμένα σε ένα domain.

**Αξίζει το domain-matched calibration;** **Κατηγορηματικά ναι καθώς εδώ είναι το σενάριο
όπου μετράει περισσότερο.** Η cross-evaluation του Task 3 είναι ξεκάθαρη: η **διαγώνιος
νικάει** (το C1 είναι το καλύτερο στα Αγγλικά, το C2 στον κώδικα), και το mismatched
calibration κοστίζει **2–3×** perplexity (61.816 του C2 στα Αγγλικά· 4.634 του C1 στον
κώδικα). Για ένα code deployment, το code-matched GPTQ calibration είναι δωρεάν
ποιότητα που θα ήταν λάθος να παραλείψουμε.

---

### Σύνδεση με το lab και την θεωρία

Κάθε task αντιστοιχεί άμεσα σε ένα πείραμα του lab (Week13 - Quantization):

- **Task 1 ↔ Part 1 (RTN σε toy matrices).** Στο lab το naive round-to-nearest σε
  συνθετικούς πίνακες έδειχνε ότι, όταν κάθε βάρος κβαντίζεται **ανεξάρτητα**, οι
  outliers υποβαθμίζουν καταστροφικά την ακρίβεια. Το ίδιο φαινόμενο μετράμε εδώ: το
  NF4 (per-weight RTN) συσσωρεύει σφάλματα rounding → **+21 %** perplexity στο INT4
  (22.664 → 27.502).
- **Task 2 ↔ Part 3 / `gptq_quantize` (TODO 9).** Ακριβώς όπως ο κανόνας του lab
  (`H = 2XXᵀ`, αντιστροφή `H_inv`, κβάντιση **στήλη-στήλη με διάδοση του σφάλματος**
  στις μη-ακόμη-κβαντισμένες στήλες), το GPTQ ελαχιστοποιεί το **σφάλμα εξόδου του
  layer** αντί του σφάλματος κάθε μεμονωμένου βάρους. Γι' αυτό **διατηρεί το σχήμα της
  κατανομής των βαρών** (βλ. `task2_weight_distributions.png`) και ανακτά **1.286 PPL
  (4.7 %)** έναντι του NF4.
- **Task 3 ↔ Extension 3 (ordering study) & «why calibration matters».** Το `H = 2XXᵀ`
  υπολογίζεται από τα **activations του calibration**. Όπως το Extension 3 έδειξε ότι η
  σειρά επεξεργασίας στηλών (που εξαρτάται από τη διαγώνιο του `H_inv`) αλλάζει το
  αποτέλεσμα, έτσι και εδώ το **domain** του calibration καθορίζει ποιο Hessian άρα και
  ποια σφάλματα αντισταθμίζονται. Ένα mismatch (C2 calibrated σε κώδικα, eval στα
  Αγγλικά) διορθώνει τα λάθος σφάλματα και η perplexity εκτοξεύεται (**61.816**).
- **Bonus A ↔ Part 2 (Hadamard rotation / QuIP# incoherence).** Στο lab ο
  μετασχηματισμός Hadamard (`W' = WH`, `x' = Hᵀx`, με `H Hᵀ = I`) μαζί με τα random
  sign diagonals του QuIP# (`D = diag(±1)`) «απλώνουν» τα βάρη και άρα μειώνουν το
  max |w| και τα κάνουν *incoherent*, ώστε το rounding να χάνει λιγότερη πληροφορία. Το
  HQQ που χρησιμοποιούμε στο Bonus A είναι **calibration-free RTN χωρίς incoherence
  rotation**, οπότε η **κατάρρευση στα 2-bit (PPL 312k)** είναι **απολύτως συνεπής με τη
  θεωρία**. Πρακτικά χωρίς τον incoherence μετασχηματισμό, μόλις 4 επίπεδα ανά βάρος δεν αρκούν
  ακριβώς το πρόβλημα που το Hadamard/QuIP# σχεδιάστηκε να λύσει. Η θεωρία προβλέπει ότι
  ένα QuIP#-style 2-bit (με Hadamard preprocessing) θα ήταν δραστικά ανθεκτικότερο από
  το naive HQQ 2-bit που μετρήσαμε.

---

### Περιορισμοί & Αναπαραγωγιμότητα

- **Hardware journey.** Ο κώδικας αναπτύχθηκε σε GTX 1660 (Turing sm_75) όπου το
  bitsandbytes INT8 (`cublasLt`) και το Marlin GPTQ kernel **δεν τρέχουν**· γι' αυτό
  τα τελικά benchmarks έγιναν σε **rented RTX 4090** (Ada sm_89), όπου η γραμμή INT8
  συμπληρώθηκε και το throughput είναι πραγματικό (`GPTQ_BACKEND=auto` → Marlin).
- **INT8 latency (χαρακτηριστικό, όχι σφάλμα).** Ακόμη και στην RTX 4090 το INT8 είναι
  αργό (10.4 tok/s) λόγω του mixed-precision outlier path του bitsandbytes που είναι βελτιστοποιημένο για μνήμη, όχι throughput· γι' αυτό στο Σενάριο 2 προτιμάμε BF16.
- **Αλλαγές dataset/backend λόγω stack.** Σε `transformers` 5.x / `datasets` 5.x:
  `wikitext`→`Salesforce/wikitext`, `codeparrot/github-code`→`mbpp`, `pg19`→
  `ML-ArXiv-Papers` (script-based → parquet)· `auto-gptq`→`gptqmodel`· HQQ μέσω native
  API. Καμία δεν αλλάζει τη μεθοδολογία απλά μόνο τα μη-συντηρούμενα entry points.
- **Μέγεθος μοντέλου.** Στο 0.5B το 4-bit ξεπερνά το budget 5 % και το 2-bit καταρρέει·
  το **scaling extension στο 1.5B** δείχνει ότι η ποινή του INT4 πέφτει (+21 %→+13.7 %)
  και το GPTQ advantage μεγαλώνει (4.7 %→5.7 %), το quantization κλιμακώνει ευνοϊκά.
- **Φορητότητα σε άλλη αρχιτεκτονική.** Το Bonus B (`collaboration_report.md`) τρέχει
  ολόκληρο το pipeline σε `meta-llama/Llama-3.2-1B-Instruct` (ξένη οικογένεια από το
  Qwen). Τα δομικά συμπεράσματα μεταφέρονται (BF16 default, INT8 ακριβές-αλλά-αργό,
  2-bit κατάρρευση, διαγώνιος Task 3), που σημαίνει ότι οι προτάσεις deployment δεν
  είναι artifacts του Qwen. Εξαίρεση-εύρημα: το ποιοτικό πλεονέκτημα του GPTQ έναντι
  NF4 είναι μοντελο-εξαρτώμενο, ενώ το πλεονέκτημα *ταχύτητας* (Marlin) είναι σταθερό.
- **Αναπαραγωγιμότητα.** Σταθερό eval corpus (`src/common.load_eval_corpus`, ίδιο σε
  όλα τα tasks)· greedy decoding· καρφωμένες εκδόσεις (`requirements.lock.txt`)· και
  `Dockerfile` (CUDA devel base με nvcc) που στήνει ολόκληρο το περιβάλλον με μία εντολή.
