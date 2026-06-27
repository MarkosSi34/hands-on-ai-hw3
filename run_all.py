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


def _namespace_outputs(task: str, model: str):
    """Append the model slug to a task's PNGs (only for non-default models)."""
    s = _slug(model)
    for f in TASK_OUTPUTS.get(task, []):
        if os.path.exists(f):
            base, ext = os.path.splitext(f)
            dst = f"{base}_{s}{ext}"
            shutil.move(f, dst)
            logging.info(f"Renamed plot → {dst} (keeping canonical names for {MODEL_NAME})")


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
    if model == MODEL_NAME:
        bonus_2bit_hqq.main()            # full FP32→INT2 curve (points are for this model)
    else:
        bonus_2bit_hqq.HQQ2BitBenchmark(model_name=model).run_pipeline()
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
        DISPATCH[t](args.model)
        if args.model != MODEL_NAME:
            _namespace_outputs(t, args.model)
    logging.info("All requested tasks finished.")


if __name__ == "__main__":
    main()
