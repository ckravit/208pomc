# plotting.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import calibration_curve

# SHAP imports are optional
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False



# ============================================================
# Helper: Ensure directory exists
# ============================================================

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)



# ============================================================
# ROC + PR Curve Plotting
# ============================================================

def plot_roc_curve(fpr, tpr, auc_value, title, output_path):
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pr_curve(precision, recall, auc_value, title, output_path):
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title}\nAP = {auc_value:.3f}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# Multi-split ROC & PR (Train/Val/Test)
# ============================================================

def plot_multi_split_roc(roc_dict, output_path):
    """
    roc_dict = {
        "train": {"fpr": ..., "tpr": ... , "auc": ...},
        "val":   {...},
        "test":  {...}
    }
    """
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))

    for split, d in roc_dict.items():
        plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{split} (AUC {d['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves (All Splits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_multi_split_pr(pr_dict, output_path):
    """
    pr_dict = {
        "train": {"precision": ..., "recall": ..., "ap": ...},
        "val":   {...},
        "test":  {...}
    }
    """
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))

    for split, d in pr_dict.items():
        plt.plot(d["recall"], d["precision"], lw=2, label=f"{split} (AP {d['ap']:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves (All Splits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# Calibration Curve
# ============================================================

def plot_calibration_curve(y_true, y_prob, title, output_path):
    _ensure_dir(output_path)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", lw=2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# Probability Histogram
# ============================================================

def plot_probability_histogram(y_true, y_prob, title, output_path):
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Negative", density=True)
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Positive", density=True)

    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# Sex-stratified ROC & PR
# ============================================================

def plot_sex_stratified_roc(sex_roc_dict, output_path):
    """
    sex_roc_dict = {
        "female": {"fpr":..., "tpr":..., "auc":...},
        "male":   {...}
    }
    """
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))

    for sex, d in sex_roc_dict.items():
        plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{sex} (AUC {d['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Sex-Stratified ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sex_stratified_pr(sex_pr_dict, output_path):
    """
    sex_pr_dict = {
        "female": {"precision":..., "recall":..., "ap":...},
        "male":   {...}
    }
    """
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))
    for sex, d in sex_pr_dict.items():
        plt.plot(d["recall"], d["precision"], lw=2, label=f"{sex} (AP {d['ap']:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Sex-Stratified PR Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# Group Bar Metrics (e.g., TP/FP/TN/FN or sensitivity/specificity per group)
# ============================================================

def plot_group_bar_metrics(group_metrics, output_path):
    """
    group_metrics = {
        "female": {"sensitivity":..., "specificity":..., ...},
        "male":   {...}
    }
    """
    _ensure_dir(output_path)

    labels = list(group_metrics.keys())
    metrics = list(next(iter(group_metrics.values())).keys())

    x = np.arange(len(labels))
    width = 0.10

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        vals = [group_metrics[g][metric] for g in labels]
        plt.bar(x + i * width, vals, width, label=metric)

    plt.xticks(x + width * len(metrics) / 2, labels)
    plt.ylabel("Value")
    plt.title("Group Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# Feature Importance (GBC)
# ============================================================

def plot_feature_importance(model, feature_names, top_n, output_path):
    _ensure_dir(output_path)

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 8))
    plt.barh(range(top_n), importances[idx][::-1])
    plt.yticks(range(top_n), np.array(feature_names)[idx][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# SHAP PLOTS
# ============================================================

def plot_shap_summary(model, X_sample, feature_names, output_path):
    if not HAS_SHAP:
        return

    _ensure_dir(output_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_shap_bar(model, X_sample, feature_names, output_path, top_n=20):
    if not HAS_SHAP:
        return

    _ensure_dir(output_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# Comparison Plots (multiple runs)
# ============================================================

def plot_compare_roc(compare_roc_dict, output_path):
    """
    compare_roc_dict = {
        "run1": {"fpr":..., "tpr":..., "auc":...},
        "run2": {...}
    }
    """
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))
    for name, d in compare_roc_dict.items():
        plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{name} (AUC {d['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Run Comparison ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_compare_pr(compare_pr_dict, output_path):
    """
    compare_pr_dict = {
        "run1": {"precision":..., "recall":..., "ap":...},
        "run2": {...}
    }
    """
    _ensure_dir(output_path)

    plt.figure(figsize=(7, 6))

    for name, d in compare_pr_dict.items():
        plt.plot(d["recall"], d["precision"], lw=2, label=f"{name} (AP {d['ap']:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Run Comparison PR Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_compare_metrics_table(compare_metrics, output_path):
    """
    compare_metrics = {
        "run1": {"auc":..., "ap":..., "brier":...},
        "run2": {...}
    }
    """
    _ensure_dir(output_path)

    runs = list(compare_metrics.keys())
    metrics = list(next(iter(compare_metrics.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 2 + len(runs) * 0.5))
    table_data = [[compare_metrics[r][m] for m in metrics] for r in runs]

    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        rowLabels=runs,
        colLabels=metrics,
        cellLoc="center",
        loc="center",
    )

    table.scale(1, 1.4)
    plt.title("Run Comparison Metrics")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# ============================================================
# High-level automatic plot generator
# ============================================================

def generate_plots(dataset_name, run_cfg, eval_results, training_metadata, logger):
    """
    Creates all standard plots for one dataset/run.
    """
    logger.info(f"Generating plots for dataset: {dataset_name}")

    out_root = training_metadata["output_dir"]
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Basic ROC / PR ----
    plot_roc_curve(
        eval_results["roc_curve"]["fpr"],
        eval_results["roc_curve"]["tpr"],
        eval_results["basic"]["roc_auc"],
        title=f"{dataset_name.upper()} Test ROC",
        output_path=os.path.join(plot_dir, "test_roc.png"),
    )

    plot_pr_curve(
        eval_results["pr_curve"]["precision"],
        eval_results["pr_curve"]["recall"],
        eval_results["basic"]["average_precision"],
        title=f"{dataset_name.upper()} Test Precision-Recall",
        output_path=os.path.join(plot_dir, "test_pr.png"),
    )

    # ---- Calibration ----
    plot_calibration_curve(
        training_metadata["y_test"],
        training_metadata["y_test_prob"],
        title=f"{dataset_name.upper()} Calibration Curve",
        output_path=os.path.join(plot_dir, "calibration.png"),
    )

    # ---- Probability Histogram ----
    plot_probability_histogram(
        training_metadata["y_test"],
        training_metadata["y_test_prob"],
        title=f"{dataset_name.upper()} Probability Histogram",
        output_path=os.path.join(plot_dir, "prob_hist.png"),
    )

    # ---- Sex-stratified ----
    sex_roc_dict = {}
    sex_pr_dict = {}

    for sex, vals in eval_results["sex_stratified"].items():
        sex_roc_dict[sex] = {
            "fpr": vals["roc_curve"]["fpr"],
            "tpr": vals["roc_curve"]["tpr"],
            "auc": vals["roc_auc"],
        }
        sex_pr_dict[sex] = {
            "precision": vals["pr_curve"]["precision"],
            "recall": vals["pr_curve"]["recall"],
            "ap": vals["average_precision"],
        }

    plot_sex_stratified_roc(sex_roc_dict, os.path.join(plot_dir, "sex_roc.png"))
    plot_sex_stratified_pr(sex_pr_dict, os.path.join(plot_dir, "sex_pr.png"))

    logger.info("Plot generation complete.")
