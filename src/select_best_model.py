from __future__ import annotations

import json
import shutil
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT / "metrics"
MODELS_DIR = ROOT / "models"

SUMMARY_PATH = METRICS_DIR / "summary_16_experiments.json"
OUTPUT_MODEL_PATH = MODELS_DIR / "best_loan_model.pkl"
OUTPUT_META_PATH = MODELS_DIR / "best_loan_model_meta.json"


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    # Load all experiment results
    runs = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    if len(runs) < 16:
        print(f"⚠️ Warning: expected 16 runs, found {len(runs)}")

    # Pick best model based on F1 score
    best = max(runs, key=lambda r: r["f1"])
    run_id = best["run_id"]

    model_path = MODELS_DIR / f"{run_id}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Copy best model
    shutil.copyfile(model_path, OUTPUT_MODEL_PATH)

    # Save metadata
    OUTPUT_META_PATH.write_text(
        json.dumps(best, indent=2),
        encoding="utf-8"
    )

    print("✅ Best loan approval model selected!")
    print("Run ID:", run_id)
    print("Best F1 score:", best["f1"])
    print("Model saved to:", OUTPUT_MODEL_PATH)
    print("Metadata saved to:", OUTPUT_META_PATH)


if __name__ == "__main__":
    main()
