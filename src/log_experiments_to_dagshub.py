import json
from pathlib import Path
import mlflow
from mlflow.exceptions import MlflowException
import os

# -------------------------------
# Configuration
# -------------------------------
DAGSHUB_USER = "RagNaidu"
DAGSHUB_REPO = "project-final"
DAGSHUB_TOKEN = "885558255579495233d71976148cec508fc62fb1"  # Use env variable in production for safety

summary_path = Path("metrics") / "summary_16_experiments.json"
if not summary_path.exists():
    raise FileNotFoundError(f"Missing: {summary_path.resolve()}")

# Use token in headers for server-side logging
tracking_uri = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("loan_prediction_16_experiments")

# -------------------------------
# Load runs
# -------------------------------
def load_runs(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
        return data["runs"]
    if isinstance(data, list):
        return data
    raise ValueError("Unknown summary JSON format. Expected a list or a dict with key 'runs'.")

runs = load_runs(summary_path)
print(f"Logging {len(runs)} runs to {tracking_uri}")

# -------------------------------
# Log each run
# -------------------------------
for r in runs:
    run_name = r.get("run_id") or r.get("name") or r.get("id") or "run"
    model_type = r.get("model_type") or r.get("model") or r.get("algo") or "unknown"
    use_pca = int(bool(r.get("use_pca", r.get("pca", False))))
    tuned = int(bool(r.get("tuned", r.get("optuna", False))))
    f1 = float(r.get("f1", r.get("f1_score", 0.0)))
    params = r.get("params", {}) or {}

    try:
        # Server-side logging; browser-safe
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("use_pca", use_pca)
            mlflow.set_tag("tuned", tuned)

            mlflow.log_param("model_type", model_type)
            mlflow.log_param("use_pca", use_pca)
            mlflow.log_param("tuned", tuned)

            for k, v in params.items():
                mlflow.log_param(str(k), str(v))

            mlflow.log_metric("f1", f1)
    except MlflowException as e:
        print(f"⚠️ Skipping run '{run_name}': {e}")

print("✅ Done. Open the DagsHub MLflow UI at:")
print(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
