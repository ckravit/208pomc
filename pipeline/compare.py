# compare.py

import os
import json
from datetime import datetime
from pathlib import Path

from evaluator import (
    load_evaluation_results,
    delong_test,
    mcnemar_test,
)

# ============================================================
# HELPERS
# ============================================================

def _ensure_list(x):
    """Ensure CLI values become a list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return x.split(",")
    return list(x)

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# DISCOVER AVAILABLE RUN FOLDERS
# ============================================================

def discover_runs(dataset_name, output_root):
    """
    Returns a list of run folders (strings) under:
        outputs/<dataset>/
    Only folders matching run_<timestamp> are returned.
    """
    dataset_dir = os.path.join(output_root, dataset_name)
    if not os.path.exists(dataset_dir):
        return []

    runs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and d.startswith("run_")
    ]
    return sorted(runs)


def autodetect_latest_run(dataset_name, output_root):
    """Return the most recent run_<timestamp> for a dataset."""
    runs = discover_runs(dataset_name, output_root)
    if not runs:
        return None
    return runs[-1]   # sorted → last = latest


# ============================================================
# LOADING ARTIFACTS FOR A RUN
# ============================================================

def load_run_artifacts(dataset_name, run_folder, output_root):
    """
    Loads:
        - evaluation.json
        - training_metadata.json
        - preds_test.npy
        - model.joblib (optional for compare)
    """
    run_dir = os.path.join(output_root, dataset_name, run_folder)

    paths = {
        "evaluation": os.path.join(run_dir, "evaluation.json"),
        "metadata": os.path.join(run_dir, "training_metadata.json"),
        "preds": os.path.join(run_dir, "preds_test.npy"),
        "model": os.path.join(run_dir, "model.joblib"),
    }

    # Required artifacts
    eval_dict = load_evaluation_results(os.path.join(run_dir))
    metadata = _load_json(paths["metadata"])

    # Preds (for McNemar)
    import numpy as np
    preds_test = np.load(paths["preds"])

    # ---- NEW: load y_test so DeLong / McNemar can work ----
    y_test_path = metadata.get("y_test_path")
    if y_test_path and os.path.exists(y_test_path):
        y_test = np.load(y_test_path)
    else:
        y_test = None

    return {
        "run_folder": run_folder,
        "eval": eval_dict,
        "metadata": metadata,
        "preds_test": preds_test,
        "y_test": y_test,
        "paths": paths,
    }


# ============================================================
# COMPARISON LOGIC
# ============================================================

def compare_runs(run_artifacts):
    """
    Input: list of dicts returned from load_run_artifacts().
    Returns a structure suitable for reporting.
    """
    comparison = {
        "runs": [],
        "pairwise_delong": [],
        "pairwise_mcnemar": []
    }

    # ------------------------------------------
    # 1. Collect metrics per run
    # ------------------------------------------
    for art in run_artifacts:
        evald = art["eval"]

        entry = {
            "run_folder": art["run_folder"],
            "roc_auc": evald["basic"]["roc_auc"],
            "average_precision": evald["basic"]["average_precision"],
            "brier": evald["brier"],
            "confusion": evald["confusion"],
            "sex": evald["sex_stratified"],
        }
        comparison["runs"].append(entry)

    # ------------------------------------------
    # 2. DeLong pairwise (AUROC) + McNemar
    # ------------------------------------------
    if len(run_artifacts) >= 2:
        for i in range(len(run_artifacts)):
            for j in range(i + 1, len(run_artifacts)):
                r1 = run_artifacts[i]
                r2 = run_artifacts[j]

                # Must have y_test for both runs
                if r1["y_test"] is not None and r2["y_test"] is not None:
                    y_true = r1["y_test"]   # both share same ground truth

                    # --- DeLong test (probabilities required) ---
                    try:
                        p_delong = delong_test(
                            y_true,
                            r1["preds_test"],   # NOT ROC-AUC; these are probabilities
                            r2["preds_test"]
                        )
                    except Exception:
                        p_delong = "Error"

                    # --- McNemar test (hard predictions required) ---
                    pred_a = (r1["preds_test"] >= 0.5).astype(int)
                    pred_b = (r2["preds_test"] >= 0.5).astype(int)

                    try:
                        p_mcnemar = mcnemar_test(y_true, pred_a, pred_b)
                    except Exception:
                        p_mcnemar = "Error"

                else:
                    p_delong = "Unavailable (no y_test)"
                    p_mcnemar = "Unavailable (no y_test)"

                # Store the results for this pair
                comparison["pairwise_delong"].append({
                    "run_a": r1["run_folder"],
                    "run_b": r2["run_folder"],
                    "p_value": p_delong,
                })

                comparison["pairwise_mcnemar"].append({
                    "run_a": r1["run_folder"],
                    "run_b": r2["run_folder"],
                    "p_value": p_mcnemar,
                })


    return comparison


# ============================================================
# BUILD MARKDOWN REPORT
# ============================================================

def build_markdown_comparison(dataset_group, comparison):
    md = []
    md.append(f"# Model Comparison Report — {dataset_group.upper()}")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ------------------------------------------
    # Per-run metrics
    # ------------------------------------------
    md.append("## Per-Run Summary\n")

    for r in comparison["runs"]:
        md.append(f"### Run: `{r['run_folder']}`")
        md.append(f"- **ROC-AUC:** {r['roc_auc']:.4f}")
        md.append(f"- **Average Precision:** {r['average_precision']:.4f}")
        md.append(f"- **Brier Score:** {r['brier']:.4f}")

        md.append("\n#### Confusion Metrics")
        conf = r["confusion"]
        for k, v in conf.items():
            md.append(f"- **{k.upper()}**: {v}")

        md.append("\n#### Sex-Stratified")
        for sex_group in ["female", "male"]:
            if sex_group in r["sex"]:
                g = r["sex"][sex_group]
                md.append(f"- **{sex_group.capitalize()}**: AUC={g['roc_auc']:.4f}, AP={g['average_precision']:.4f}")

        md.append("\n")

    # ------------------------------------------
    # Pairwise Statistical Tests
    # ------------------------------------------
    md.append("\n## Pairwise Statistical Tests\n")

    md.append("### DeLong (AUROC difference)")
    for row in comparison["pairwise_delong"]:
        md.append(f"- **{row['run_a']} vs {row['run_b']}**: p={row['p_value']}")

    md.append("\n### McNemar (classification differences)")
    for row in comparison["pairwise_mcnemar"]:
        md.append(f"- **{row['run_a']} vs {row['run_b']}**: p={row['p_value']}")

    return "\n".join(md)


# ============================================================
# SAVE REPORT
# ============================================================

def save_comparison_report(dataset_group, content, output_root):
    compare_dir = os.path.join(output_root, "compare", dataset_group)
    os.makedirs(compare_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(compare_dir, f"compare_{timestamp}.md")

    with open(path, "w") as f:
        f.write(content)

    return path


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def run_comparison(dataset_name_arg, run_cfg, compare_targets, logger):
    """
    dataset_name_arg: "abp", "ppg", or "abp,ppg"
    compare_targets: list of run folders OR ["latest"]
    """
    logger.info("Running comparison...")

    dataset_list = _ensure_list(dataset_name_arg)
    output_root = run_cfg["paths"]["output_root"]

    # ============ Determine which runs to load ============
    run_inputs = {}
    for ds in dataset_list:

        if compare_targets == ["latest"]:
            latest = autodetect_latest_run(ds, output_root)
            if latest is None:
                logger.warning(f"No runs for dataset {ds}")
                continue
            run_inputs[ds] = [latest]
        else:
            run_inputs[ds] = compare_targets

    # ============ Load artifacts ============
    results_by_dataset = {}
    for ds, run_list in run_inputs.items():
        artifacts = []
        for r in run_list:
            artifacts.append(
                load_run_artifacts(ds, r, output_root)
            )
        results_by_dataset[ds] = artifacts

    # ============ Compare each dataset separately ============
    compare_paths = []

    for ds, artifacts in results_by_dataset.items():
        comparison = compare_runs(artifacts)
        content = build_markdown_comparison(ds, comparison)
        path = save_comparison_report(ds, content, output_root)
        compare_paths.append(path)
        logger.info(f"Saved comparison for {ds} → {path}")

    return compare_paths
