# evaluator.py

import os
import json
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import norm


# ============================================================
# BRIER SCORE
# ============================================================

def compute_brier_score(y_true, y_prob):
    """Compute the Brier score (mean squared probability error)."""
    return float(brier_score_loss(y_true, y_prob))


# ============================================================
# BASIC METRICS (ROC-AUC, AP)
# ============================================================

def compute_basic_metrics(y_true, y_prob):
    """Return ROC-AUC and Average Precision."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


# ============================================================
# CONFUSION METRICS (Spec, Sens, PPV, NPV)
# ============================================================

def compute_confusion_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute confusion-matrix derived metrics:
    Sensitivity, Specificity, PPV, NPV.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan,
    }


# ============================================================
# ROC & PR CURVE DATA (fed to plotting.py)
# ============================================================

def compute_roc_curve(y_true, y_prob):
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresh.tolist(),
    }


def compute_pr_curve(y_true, y_prob):
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresh.tolist(),
    }


# ============================================================
# DELONG TEST – AUROC comparison
# ============================================================

def delong_test(y_true, prob_a, prob_b):
    """
    Two-model DeLong test for AUROC difference.
    Returns p-value.
    """
    # Helper functions
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N - 1 and Z[j] == Z[j + 1]:
                j += 1
            T[i:j+1] = 0.5 * (i + j) + 1
            i = j + 1
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    def fastDeLong(preds_sorted, label_1_count):
        m = label_1_count
        n = preds_sorted.shape[1] - m
        pos = preds_sorted[:, :m]
        neg = preds_sorted[:, m:]

        k = preds_sorted.shape[0]
        Tx = np.zeros((k, m))
        Ty = np.zeros((k, n))

        for r in range(k):
            Tx[r] = compute_midrank(pos[r])
            Ty[r] = compute_midrank(neg[r])

        aucs = np.mean(Tx, axis=1) / m - np.mean(Ty, axis=1) / n
        v01 = np.cov(Tx) / m
        v10 = np.cov(Ty) / n

        return aucs, v01 + v10

    y_true = np.array(y_true)
    order = np.argsort(-y_true)
    preds = np.vstack([prob_a, prob_b])[:, order]
    y_sorted = y_true[order]

    label_1_count = int(y_sorted.sum())

    aucs, cov = fastDeLong(preds, label_1_count)
    delta = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]

    z = delta / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))

    return float(p)


# ============================================================
# MCNEMAR TEST
# ============================================================

def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar’s test for 2 classifiers.
    Hard predictions required (0/1).
    Returns p-value.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_wrong_b_correct = np.sum((pred_a != y_true) & (pred_b == y_true))
    b_wrong_a_correct = np.sum((pred_b != y_true) & (pred_a == y_true))

    table = [[0, a_wrong_b_correct],
             [b_wrong_a_correct, 0]]

    result = mcnemar(table, exact=False, correction=True)
    return float(result.pvalue)


# ============================================================
# SEX-STRATIFIED EVALUATION
# ============================================================

def evaluate_by_sex(y_true, y_prob, sex_vector, threshold=0.5):
    """
    Compute ROC-AUC, AP, and confusion metrics separately
    for male (1) and female (0).
    """
    results = {}
    for label, name in [(0, "female"), (1, "male")]:
        idx = np.where(sex_vector == label)[0]
        if len(idx) == 0:
            continue

        y_t = y_true[idx]
        y_p = y_prob[idx]

        results[name] = {
            "n": int(len(idx)),
            "roc_auc": float(roc_auc_score(y_t, y_p)),
            "average_precision": float(average_precision_score(y_t, y_p)),
            **compute_confusion_metrics(y_t, y_p, threshold),
        }

    return results


# ============================================================
# FULL EVALUATION PIPELINE
# ============================================================

def evaluate_model(predictions, y_true_splits, sex_splits, threshold=0.5):
    """
    predictions must contain:
        - pred_test (probabilities)
        - pred_test_bin (hard predictions)
    """
    y_test = y_true_splits["test"]
    sex_test = sex_splits["test"]

    prob_test = predictions["pred_test"]
    pred_test = predictions["pred_test_bin"]

    return {
        "brier": compute_brier_score(y_test, prob_test),
        "basic": compute_basic_metrics(y_test, prob_test),
        "confusion": compute_confusion_metrics(y_test, prob_test, threshold),
        "roc_curve": compute_roc_curve(y_test, prob_test),
        "pr_curve": compute_pr_curve(y_test, prob_test),
        "sex_stratified": evaluate_by_sex(y_test, prob_test, sex_test, threshold),
    }


# ============================================================
# SAVING / LOADING EVALUATION ARTIFACTS
# ============================================================

def save_evaluation_results(eval_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "evaluation.json")

    # Ensure everything is JSON-serializable
    safe_dict = json.loads(json.dumps(eval_dict, default=lambda x: float(x)))

    with open(out_path, "w") as f:
        json.dump(safe_dict, f, indent=2)

    return out_path


def load_evaluation_results(output_dir):
    path = os.path.join(output_dir, "evaluation.json")
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# MAIN WRAPPER
# ============================================================

def run_evaluation(dataset_name, run_cfg, model_results, logger):
    """
    Called by main.py after training is complete.
    """
    logger.info(f"Running evaluation for dataset: {dataset_name}")

    y_true_splits = {"test": model_results["y_test"]}
    sex_splits = {"test": model_results["sex_test"]}

    predictions = {
        "pred_test": model_results["pred_test"],
        "pred_test_bin": model_results["pred_test_bin"],
    }

    eval_dict = evaluate_model(predictions, y_true_splits, sex_splits)

    logger.info("Evaluation completed.")
    return eval_dict
