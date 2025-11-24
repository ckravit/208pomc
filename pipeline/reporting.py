# reporting.py

import os
from datetime import datetime


# ============================================================
# Helpers
# ============================================================

def _section(title):
    return f"\n## {title}\n"


def _fmt_metric(name, value, indent=0):
    pad = " " * indent
    if isinstance(value, float):
        value = f"{value:.4f}"
    return f"{pad}- **{name}**: {value}\n"


# ============================================================
# MARKDOWN REPORT
# ============================================================

def build_markdown_report(dataset_name, eval_results, training_metadata, preprocess_meta):
    md = []
    md.append(f"# Model Evaluation Report — {dataset_name.upper()}")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # -------------------------------------
    # Preprocessing Summary
    # -------------------------------------
    md.append(_section("Preprocessing Summary"))
    md.append(_fmt_metric("Total samples", preprocess_meta.get("n_total", "N/A")))
    md.append(_fmt_metric("Train size", preprocess_meta.get("n_train", "N/A")))
    md.append(_fmt_metric("Validation size", preprocess_meta.get("n_val", "N/A")))
    md.append(_fmt_metric("Test size", preprocess_meta.get("n_test", "N/A")))
    md.append(_fmt_metric("Imputer type", preprocess_meta.get("imputer_type", "N/A")))
    md.append("\n")

    # -------------------------------------
    # Core Metrics
    # -------------------------------------
    md.append(_section("Core Metrics (Test Set)"))
    md.append(_fmt_metric("ROC-AUC", eval_results["basic"]["roc_auc"]))
    md.append(_fmt_metric("Average Precision", eval_results["basic"]["average_precision"]))
    md.append(_fmt_metric("Brier Score", eval_results["brier"]))

    # Confusion metrics
    md.append("\n### Confusion-Derived Metrics\n")
    conf = eval_results["confusion"]
    for k in ["sensitivity", "specificity", "ppv", "npv", "TP", "FP", "TN", "FN"]:
        if k in conf:
            md.append(_fmt_metric(k.upper(), conf[k]))

    # -------------------------------------
    # Sex-Stratified Metrics
    # -------------------------------------
    md.append(_section("Sex-Stratified Performance"))
    sex = eval_results["sex_stratified"]

    for group in ["female", "male"]:
        if group in sex:
            vals = sex[group]
            md.append(f"\n### {group.capitalize()}\n")
            md.append(_fmt_metric("n", vals["n"]))
            md.append(_fmt_metric("ROC-AUC", vals["roc_auc"]))
            md.append(_fmt_metric("Average Precision", vals["average_precision"]))
            md.append(_fmt_metric("Sensitivity", vals["sensitivity"]))
            md.append(_fmt_metric("Specificity", vals["specificity"]))
            md.append(_fmt_metric("PPV", vals["ppv"]))
            md.append(_fmt_metric("NPV", vals["npv"]))

    # -------------------------------------
    # Training Metadata
    # -------------------------------------
    md.append(_section("Training Metadata"))
    for k, v in training_metadata.items():
        if isinstance(v, dict):
            md.append(f"\n**{k}**:\n")
            for kk, vv in v.items():
                md.append(_fmt_metric(kk, vv, indent=2))
        else:
            md.append(_fmt_metric(k, v))

    return "\n".join(md)


# ============================================================
# TEXT REPORT
# ============================================================

def build_text_report(dataset_name, eval_results, training_metadata, preprocess_meta):
    lines = []
    lines.append(f"MODEL EVALUATION REPORT — {dataset_name.upper()}")
    lines.append("=" * 60)

    lines.append("\nPreprocessing Summary:")
    lines.append(f"  Total samples: {preprocess_meta.get('n_total')}")
    lines.append(f"  Train/Val/Test: {preprocess_meta.get('n_train')}/"
                 f"{preprocess_meta.get('n_val')}/"
                 f"{preprocess_meta.get('n_test')}")
    lines.append(f"  Imputer: {preprocess_meta.get('imputer_type')}")

    lines.append("\nCore Test Metrics:")
    lines.append(f"  ROC-AUC: {eval_results['basic']['roc_auc']:.4f}")
    lines.append(f"  Average Precision: {eval_results['basic']['average_precision']:.4f}")
    lines.append(f"  Brier Score: {eval_results['brier']:.4f}")

    conf = eval_results["confusion"]
    lines.append("\nConfusion Metrics:")
    for k, v in conf.items():
        lines.append(f"  {k.upper()}: {v}")

    lines.append("\nSex-Stratified Performance:")
    sex = eval_results["sex_stratified"]
    for group in ["female", "male"]:
        if group in sex:
            g = sex[group]
            lines.append(f"  {group.capitalize()} (n={g['n']})")
            lines.append(f"    ROC-AUC: {g['roc_auc']:.4f}")
            lines.append(f"    AP: {g['average_precision']:.4f}")
            lines.append(f"    Sens: {g['sensitivity']:.4f}")
            lines.append(f"    Spec: {g['specificity']:.4f}")

    lines.append("\nTraining Metadata:")
    for k, v in training_metadata.items():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(content, output_dir, format_name):
    os.makedirs(output_dir, exist_ok=True)

    ext = "md" if format_name == "markdown" else "txt"
    path = os.path.join(output_dir, f"report.{ext}")

    with open(path, "w") as f:
        f.write(content)

    return path


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def generate_report(dataset_name, run_cfg, eval_results, training_metadata, preprocess_meta, run_folder, logger):
    """
    Called by main.py after training + evaluation.
    run_folder is provided by main.py, not by training_metadata.
    """
    fmt = run_cfg.get("report_format", "markdown").lower()
    logger.info(f"Generating {fmt} report for dataset {dataset_name}...")

    # Choose format
    if fmt == "markdown":
        content = build_markdown_report(dataset_name, eval_results, training_metadata, preprocess_meta)
    else:
        content = build_text_report(dataset_name, eval_results, training_metadata, preprocess_meta)

    # Save to correct run folder
    output_root = run_cfg["paths"]["output_root"]
    output_dir = os.path.join(output_root, dataset_name, run_folder)

    path = save_report(content, output_dir, fmt)

    logger.info(f"Report saved to {path}")
    return path
