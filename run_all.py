"""
run_all.py — μοναδικό entry point για όλο το pipeline (GPU-session driver).

Παραδείγματα:
    python run_all.py --tasks all
    python run_all.py --tasks 1 2 3 bonus ablation
    python run_all.py --tasks 1 2 --model Qwen/Qwen2.5-1.5B   # scaling extension

Τα GPTQ caches γίνονται namespaced ανά μοντέλο (gptq_*_<slug>), ώστε το --model
να μην ξαναχρησιμοποιεί ποτέ quantized weights άλλου μοντέλου.

Inference kernel του GPTQ: env var GPTQ_BACKEND (default TORCH· βάλε AUTO σε
Ampere+ για το γρήγορο Marlin path):
    GPTQ_BACKEND=auto python run_all.py --tasks 2 3
"""
import argparse
import logging
import os
import shutil

from src.common import MODEL_NAME
import task1_benchmark
import task2_gptq
import task3_calibration
import bonus_2bit_hqq
import ablation_gptq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

ALL_TASKS = ["1", "2", "3", "bonus", "ablation"]


TASK_OUTPUTS = {
    "1": ["results/task1_benchmarks.png"],
    "2": ["results/task2_weight_distributions.png", "results/task2_gptq_comparison.png"],
    "3": ["results/task3_calibration_study.png"],
    "bonus": ["results/bonus_2bit_precision_curve.png"],
    "ablation": ["results/ablation_gptq.png"],
}


def _slug(model: str):
    return model.split("/")[-1].replace(".", "_").replace("-", "_")


def _backup_canonical(task: str) -> dict:
    """Copy existing canonical PNGs aside so a non-default-model run can't clobber
    them when its task overwrites the fixed plot paths."""
    saved = {}
    for f in TASK_OUTPUTS.get(task, []):
        if os.path.exists(f):
            saved[f] = f + ".canonbak"
            shutil.copy2(f, saved[f])
    return saved


def _namespace_outputs(task: str, model: str, saved: dict):
    """Move the just-generated (non-default) PNGs to model-slug names, then restore
    the backed-up canonical (default-model) plots."""
    s = _slug(model)
    for f in TASK_OUTPUTS.get(task, []):
        if os.path.exists(f):
            base, ext = os.path.splitext(f)
            shutil.move(f, f"{base}_{s}{ext}")
            logging.info(f"Saved model-specific plot → {base}_{s}{ext}")
    for f, bak in saved.items():
        shutil.move(bak, f)
        logging.info(f"Restored canonical plot {f} (for {MODEL_NAME})")


def run_task1(model):
    task1_benchmark.PrecisionBenchmark(model_name=model).run_pipeline()


def run_task2(model):
    task2_gptq.GPTQ_DIR = f"gptq_model_{_slug(model)}"
    task2_gptq.GPTQComparison(model_name=model).run_pipeline()


def run_task3(model):
    s = _slug(model)
    task3_calibration.RUNS = [
        ("C1", "Wikipedia", f"gptq_C1_{s}"),
        ("C2", "Python",    f"gptq_C2_{s}"),
        ("C3", "Academic",  f"gptq_C3_{s}"),
    ]
    task3_calibration.CalibrationStudy(model_name=model).run_pipeline()


def run_bonus(model):
    # NB: call the pipeline directly, NOT bonus_2bit_hqq.main() — main() re-parses
    # sys.argv and would choke on run_all's own flags (e.g. --tasks all).
    hqq_row = bonus_2bit_hqq.HQQ2BitBenchmark(model_name=model).run_pipeline()
    if model == MODEL_NAME:
        points = {
            "FP32": (1884.6, 22.733),
            "BF16": (942.3, 22.664),
            "INT8": (601.0, 22.936),
            "INT4": (430.4, 27.502),
            "INT2": (hqq_row["Size (MB)"], hqq_row["Perplexity"]),
        }
        bonus_2bit_hqq.plot_precision_curve(points)
    else:
        logging.info("Bonus curve skipped for non-default model (Task 1 points differ).")


def run_ablation(model):
    ablation_gptq.run_ablation(model_name=model)


DISPATCH = {
    "1": run_task1, "2": run_task2, "3": run_task3,
    "bonus": run_bonus, "ablation": run_ablation,
}


def main():
    parser = argparse.ArgumentParser(description="HW3 — full pipeline driver")
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        help="Υποσύνολο από {1,2,3,bonus,ablation} ή 'all'.")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model id.")
    args = parser.parse_args()

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks
    logging.info(f"Running tasks {tasks} on model '{args.model}'")
    for t in tasks:
        if t not in DISPATCH:
            logging.warning(f"Unknown task '{t}', skipping.")
            continue
        logging.info(f"================  TASK {t}  ================")
        if args.model != MODEL_NAME:
            saved = _backup_canonical(t)
            DISPATCH[t](args.model)
            _namespace_outputs(t, args.model, saved)
        else:
            DISPATCH[t](args.model)
    logging.info("All requested tasks finished.")


if __name__ == "__main__":
    main()
