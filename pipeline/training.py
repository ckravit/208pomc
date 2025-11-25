# training.py

import os
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# Build the base model (currently only GBC)
# ============================================================
def build_model(model_name, model_cfg, dataset_name, logger):
    """
    Create the correct model object from model_config.yaml settings.
    Only GBC is supported for now.
    """
    if model_name.lower() != "gbc":
        raise ValueError(f"Unsupported model: {model_name}")

    # NOTE: We DO NOT set hyperparameters here.
    # Those come from the param_grid inside GridSearchCV.
    from sklearn.ensemble import GradientBoostingClassifier
    logger.info(f"[TRAIN] Building GradientBoostingClassifier for {dataset_name}")
    return GradientBoostingClassifier(random_state=model_cfg.get("seed", 42))



# ============================================================
# Hyperparameter Search
# ============================================================
def run_hyperparameter_search(model, param_grid, X_train, y_train, strat_train, cv_cfg, scoring_cfg, logger):
    """
    Execute stratified CV hyperparameter search using the user-specified param_grid.
    Returns:
        grid_search : fitted GridSearchCV object
        cv_results_df : DataFrame of cv_results_
    """
    logger.info("[TRAIN] Starting hyperparameter search")

    # Stratified CV object using YOUR combined sex+POMC labels
    skf = StratifiedKFold(
        n_splits=cv_cfg["folds"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg.get("seed", 42),
    )

    # Build GridSearch
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=[scoring_cfg["primary"], scoring_cfg["secondary"]],
        refit=scoring_cfg["primary"],   # final model optimized on AP
        cv=skf.split(X_train, strat_train),
        return_train_score=True,
        n_jobs=-1,
        verbose=0,
    )

    # Fit
    grid.fit(X_train, y_train)
    logger.info("[TRAIN] Hyperparameter search complete")

    # Save CV results
    cv_results_df = pd.DataFrame(grid.cv_results_)
    return grid, cv_results_df



# ============================================================
# Fit final model using best params
# ============================================================
def fit_final_model(model, X_train, y_train, best_params, logger):
    """
    Rebuild model using best_params and fit on full training set.
    """
    logger.info(f"[TRAIN] Fitting final model with params: {best_params}")

    from sklearn.ensemble import GradientBoostingClassifier

    final_model = GradientBoostingClassifier(
        **best_params,
        random_state=best_params.get("random_state", 42),
    )
    final_model.fit(X_train, y_train)
    return final_model



# ============================================================
# Predict on train/val/test
# ============================================================
def predict_all_sets(model, X_train, X_val, X_test, logger):
    """
    Compute probability predictions for all sets.
    """
    logger.info("[TRAIN] Generating predictions for train/val/test")

    pred_train = model.predict_proba(X_train)[:, -1]
    pred_val   = model.predict_proba(X_val)[:, -1]
    pred_test  = model.predict_proba(X_test)[:, -1]

    pred_test_bin = (pred_test >= 0.5).astype(int)

    return pred_train, pred_val, pred_test, pred_test_bin



# ============================================================
# Save model + predictions + metadata
# ============================================================
def save_trained_model(model, output_dir):
    path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, path)
    return path


def save_predictions(predictions, output_dir):
    path = os.path.join(output_dir, "y_pred_test.npy")
    np.save(path, predictions)
    return path


def save_training_metadata(metadata, output_dir):
    path = os.path.join(output_dir, "training_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path



# ============================================================
# Main Entry Point for Training
# ============================================================
def train_model(dataset_name, run_cfg, model_cfg, preprocess_pkg, logger):
    """
    Full training pipeline:
        - build model
        - run hyperparameter search
        - fit final model
        - generate predictions
        - save artifacts
        - return training_pkg for evaluator/plotting

    Inputs:
        dataset_name  ("abp" or "ppg")
        run_cfg       parsed run_config.yaml
        model_cfg     parsed model_config.yaml
        preprocess_pkg dictionary returned by preprocess_dataset()
        logger        pipeline logger
    """

    logger.info(f"\n========== TRAINING: {dataset_name.upper()} ==========")

    # Build new run directory path
    run_timestamp = datetime.now().strftime(run_cfg["timestamp_format"])
    output_dir = os.path.join(run_cfg["paths"]["output_root"],
                            dataset_name,
                            run_cfg["folder_naming"]["model_folder"],
                            f"run_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[TRAIN] Saving outputs to: {output_dir}")
    
    # -----------------------------------------------------------
    # Extract split data from preprocess_pkg
    # -----------------------------------------------------------
    X_train = preprocess_pkg["X_train"]
    y_train = preprocess_pkg["y_train"]
    strat_train = preprocess_pkg["strat_train"]

    X_val = preprocess_pkg["X_val"]
    y_val = preprocess_pkg["y_val"]

    X_test = preprocess_pkg["X_test"]
    y_test = preprocess_pkg["y_test"]
    sex_test = preprocess_pkg["sex_test"]   # needed for sex-stratified eval


    # -----------------------------------------------------------
    # Build model core
    # -----------------------------------------------------------
    model = build_model(
        model_name=model_cfg["model_name"],
        model_cfg=model_cfg,
        dataset_name=dataset_name,
        logger=logger
    )

    param_grid = model_cfg["param_grid"][dataset_name]
    cv_cfg = model_cfg["cv"]
    scoring_cfg = model_cfg["scoring"]


    # -----------------------------------------------------------
    # Hyperparameter search
    # -----------------------------------------------------------
    grid_search, cv_df = run_hyperparameter_search(
        model=model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        strat_train=strat_train,
        cv_cfg=cv_cfg,
        scoring_cfg=scoring_cfg,
        logger=logger,
    )


    # Identify best parameter set
    best_params = grid_search.best_params_
    logger.info(f"[TRAIN] Best parameters: {best_params}")

    # -----------------------------------------------------------
    # Fit final model
    # -----------------------------------------------------------
    final_model = fit_final_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        best_params=best_params,
        logger=logger,
    )


    # -----------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------
    pred_train, pred_val, pred_test, pred_test_bin = \
        predict_all_sets(final_model, X_train, X_val, X_test, logger)


    # -----------------------------------------------------------
    # Save model, predictions, metadata
    # -----------------------------------------------------------

    # Save model
    model_path = save_trained_model(final_model, output_dir)

    # Save test-set probability predictions
    preds_path = save_predictions(pred_test, output_dir)

    # ---- NEW: Save y_test ----
    y_test_path = os.path.join(output_dir, "y_test.npy")
    np.save(y_test_path, y_test)

    # Build metadata
    metadata = {
        "dataset": dataset_name,
        "model_type": model_cfg["model_name"],
        "best_params": best_params,
        "cv_results_shape": cv_df.shape,
        "model_path": model_path,
        "predictions_path": preds_path,

        # ---- NEW: reference so compare.py can load y_test ----
        "y_test_path": y_test_path,
        # "preprocess_folder": f'preprocess_{preprocess_pkg["metadata"]["preprocess_timestamp"]}',
        "preprocess_folder": preprocess_pkg["output_dir"]
    }

    metadata_path = save_training_metadata(metadata, output_dir)

    logger.info("[TRAIN] Artifacts saved")


    # -----------------------------------------------------------
    # Final return package
    # -----------------------------------------------------------
    training_pkg = {
        "dataset": dataset_name,
        "model": final_model,
        "cv_results": cv_df,
        "best_params": best_params,

        "y_pred_train": pred_train,
        "y_pred_val": pred_val,
        "y_pred_test": pred_test,
        "y_pred_test_bin": pred_test_bin,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "sex_test": sex_test,

        "output_dir": output_dir,
        "metadata_path": metadata_path,
        "model_path": model_path,
    }

    logger.info(f"[TRAIN] Completed training for {dataset_name.upper()}")

    return training_pkg
