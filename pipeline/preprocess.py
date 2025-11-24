"""
preprocess.py
Modular preprocessing pipeline for ABP/PPG datasets using Polars + NumPy.

This module:
- Loads feature + outcome CSVs
- Cleans and encodes the sex column
- Joins outcomes to features
- Generates stratification labels
- Performs a 70/10/20 stratified split
- Fits/applies an imputer
- Saves minimal artifacts:
      splits.json
      feature_names.json
      imputer.joblib
      metadata.yaml
- Supports loading existing preprocessing artifacts

Public functions:
    preprocess_dataset(dataset_name, run_cfg, logger)
    load_existing_preprocess(dataset_name, run_cfg, logger)

All other functions are private helpers (underscored).
"""

import os
import json
import yaml
import numpy as np
import polars as pl
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from datetime import datetime


# ============================================================
# Internal: data cleaning helpers
# ============================================================

def _clean_sex_column(X: pl.DataFrame, logger):
    """
    Lowercase, strip, normalize sex column.
    """
    logger.debug("Cleaning sex column: lowercase + strip + normalize.")

    X = X.with_columns(
        pl.col("sex")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.strip()
        .alias("sex")
    )

    # Normalize known variants
    replacements = {
        "m": "male",
        "f": "female",
        "unknown": "unknown",
        "": "unknown",
    }

    X = X.with_columns(
        pl.when(pl.col("sex").is_in(list(replacements.keys())))
        .then(pl.col("sex").replace_strict(replacements))
        .otherwise(pl.col("sex"))
        .alias("sex")
    )

    # Log distribution
    vc = X["sex"].value_counts()
    logger.debug(f"Sex distribution after initial cleaning:\n{vc}")

    return X


def _compute_sex_majority(X: pl.DataFrame, logger) -> str:
    """
    Determine majority sex among known values (male/female only).
    """
    logger.debug("Computing majority sex among known values.")

    temp = X.filter(pl.col("sex") != "unknown")
    mode_res = temp["sex"].mode()
    majority = mode_res[0] if len(mode_res) > 0 else "male"  # fallback

    logger.debug(f"Majority sex determined to be: {majority}")
    return majority


def _encode_sex_binary(X: pl.DataFrame, majority_sex: str, logger) -> pl.DataFrame:
    """
    Replace unknown with majority, then encode male=1, female=0.
    """
    logger.debug("Replacing 'unknown' with majority sex and encoding binary.")

    X = X.with_columns(
        pl.when(pl.col("sex") == "unknown")
        .then(majority_sex)
        .otherwise(pl.col("sex"))
        .alias("sex")
    )

    X = X.with_columns(
        (pl.col("sex") == "male").cast(pl.Int8).alias("sex_binary")
    )

    vc = X["sex_binary"].value_counts()
    logger.debug(f"Sex binary distribution:\n{vc}")

    return X


# ============================================================
# Internal: stratification helpers
# ============================================================

def _make_strat_labels(sex_binary: np.ndarray, pomc: np.ndarray, logger) -> np.ndarray:
    """
    Create labels like '0_1' (sex, pomc) for stratified splitting.
    """
    logger.debug("Building combined stratification labels (sex_pomc).")

    labels = np.array([f"{s}_{y}" for s, y in zip(sex_binary, pomc)], dtype=str)

    unique, counts = np.unique(labels, return_counts=True)
    logger.debug("Stratification label counts:")
    for u, c in zip(unique, counts):
        logger.debug(f"  {u}: {c}")

    return labels


# ============================================================
# Internal: splitting logic
# ============================================================

def _stratified_split_70_10_20(X_np, y_np, strat_labels, seed, logger):
    """
    Perform a 70/10/20 stratified split using two-stage train_test_split.
    """
    logger.debug("Performing stratified 70/10/20 split.")

    train_idx, temp_idx = train_test_split(
        np.arange(len(X_np)),
        test_size=0.30,
        random_state=seed,
        stratify=strat_labels
    )

    val_rel = 1.0 / 3.0  # of the remaining 30%
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_rel,
        random_state=seed,
        stratify=strat_labels[temp_idx]
    )

    logger.debug(f"Train size: {len(train_idx)}")
    logger.debug(f"Val size:   {len(val_idx)}")
    logger.debug(f"Test size:  {len(test_idx)}")

    # Overlap check
    if (
        set(train_idx) & set(val_idx)
        or set(train_idx) & set(test_idx)
        or set(val_idx) & set(test_idx)
    ):
        logger.error("ERROR: Overlap detected in split indices!")
        raise RuntimeError("Overlap detected in split indices.")

    return train_idx, val_idx, test_idx


# ============================================================
# Internal: imputer helpers
# ============================================================

def _fit_imputer(X_train_np, strategy, logger):
    logger.debug(f"Fitting imputer with strategy = {strategy}.")
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_train_np)
    return imputer


def _apply_imputer(imputer, X_np, logger):
    logger.debug("Applying imputer to dataset.")
    return imputer.transform(X_np)


# ============================================================
# Internal: saving + loading artifacts
# ============================================================

def _save_splits(train_idx, val_idx, test_idx, outdir):
    data = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    with open(os.path.join(outdir, "splits.json"), "w") as f:
        json.dump(data, f, indent=2)


def _save_feature_names(feature_names, outdir):
    with open(os.path.join(outdir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)


def _save_imputer(imputer, outdir):
    joblib.dump(imputer, os.path.join(outdir, "imputer.joblib"))


def _save_preprocess_metadata(metadata: dict, outdir):
    with open(os.path.join(outdir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)


def _load_splits(outdir):
    with open(os.path.join(outdir, "splits.json"), "r") as f:
        return json.load(f)


def _load_feature_names(outdir):
    with open(os.path.join(outdir, "feature_names.json"), "r") as f:
        return json.load(f)


def _load_imputer(outdir):
    return joblib.load(os.path.join(outdir, "imputer.joblib"))


def _load_preprocess_metadata(outdir):
    with open(os.path.join(outdir, "metadata.yaml"), "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Public: load existing preprocess artifacts
# ============================================================

def load_existing_preprocess(dataset_name, run_cfg, logger):
    """
    Load splits, feature names, imputer, metadata from last preprocess_<timestamp> folder.
    """
    out_root = run_cfg["paths"]["output_root"]
    model_folder = run_cfg["folder_naming"]["model_folder"]
    dataset_dir = os.path.join(out_root, dataset_name, model_folder)

    # pick most recent preprocess_* folder
    subdirs = [
        d for d in os.listdir(dataset_dir)
        if d.startswith("preprocess_")
    ]
    if not subdirs:
        raise FileNotFoundError("No preprocess_* directory found.")

    latest = sorted(subdirs)[-1]
    preprocess_dir = os.path.join(dataset_dir, latest)

    logger.info(f"Loading existing preprocess artifacts from: {preprocess_dir}")

    splits = _load_splits(preprocess_dir)
    feature_names = _load_feature_names(preprocess_dir)
    imputer = _load_imputer(preprocess_dir)
    metadata = _load_preprocess_metadata(preprocess_dir)

    return {
        "splits": splits,
        "feature_names": feature_names,
        "imputer": imputer,
        "metadata": metadata,
        "preprocess_dir": preprocess_dir,
    }


# ============================================================
# Public: main preprocessing pipeline
# ============================================================

def preprocess_dataset(dataset_name, run_cfg, logger):
    """
    Main entry point:
        - load files
        - clean sex
        - join outcomes
        - stratify + split
        - impute
        - save all artifacts
        - return numpy arrays + artifacts
    """

    logger.info(f"Starting preprocessing for dataset: {dataset_name}")

    # --------------------------------------------------------
    # Paths + filenames
    # --------------------------------------------------------
    in_root = run_cfg["paths"]["input_root"]
    out_root = run_cfg["paths"]["output_root"]
    model_folder = run_cfg["folder_naming"]["model_folder"]

    feature_file = run_cfg["input_files"][dataset_name]["features"]
    outcome_file = run_cfg["input_files"][dataset_name]["outcomes"]

    feature_path = os.path.join(in_root, feature_file)
    outcome_path = os.path.join(in_root, outcome_file)

    # --------------------------------------------------------
    # Load CSV files
    # --------------------------------------------------------
    logger.debug(f"Reading feature file: {feature_path}")
    X = pl.read_csv(feature_path)

    logger.debug(f"Reading outcomes file: {outcome_path}")
    Y = pl.read_csv(outcome_path).rename({"StudyID": "Study_ID"})

    # --------------------------------------------------------
    # Fix Study_ID dtype alignment
    # --------------------------------------------------------
    logger.debug("Casting Study_ID to Int64 in both tables.")
    X = X.with_columns(pl.col("Study_ID").cast(pl.Int64))
    Y = Y.with_columns(pl.col("Study_ID").cast(pl.Int64))

    # --------------------------------------------------------
    # Join outcomes
    # --------------------------------------------------------
    logger.debug("Joining outcomes onto feature table.")
    X = X.join(Y.select(["Study_ID", "POMC"]), on="Study_ID", how="left")

    if X["POMC"].null_count() > 0:
        logger.warning("Rows found with missing POMC after join.")

    # --------------------------------------------------------
    # Clean and encode sex
    # --------------------------------------------------------
    X = _clean_sex_column(X, logger)
    majority = _compute_sex_majority(X, logger)
    X = _encode_sex_binary(X, majority, logger)

    # --------------------------------------------------------
    # Extract arrays for splitting + training
    # --------------------------------------------------------
    y_np = X["POMC"].to_numpy()
    sex_np = X["sex_binary"].to_numpy()

    # Drop non-feature columns (Study_ID, POMC, sex, sex_binary)
    drop_cols = ["Study_ID", "POMC", "sex", "sex_binary"]
    feature_cols = [c for c in X.columns if c not in drop_cols]

    X_np = X.select(feature_cols).to_numpy()

    logger.debug(f"Extracted {len(feature_cols)} feature columns.")

    # --------------------------------------------------------
    # Make stratification labels
    # --------------------------------------------------------
    strat_labels = _make_strat_labels(sex_np, y_np, logger)

    # --------------------------------------------------------
    # Do the stratified 70/10/20 split
    # --------------------------------------------------------
    seed = run_cfg["seed"]
    train_idx, val_idx, test_idx = _stratified_split_70_10_20(
        X_np, y_np, strat_labels, seed, logger
    )

    # --------------------------------------------------------
    # Fit imputer + transform
    # --------------------------------------------------------
    imputer_strategy = run_cfg["imputer"]["strategy"]
    imputer = _fit_imputer(X_np[train_idx], imputer_strategy, logger)

    X_train = _apply_imputer(imputer, X_np[train_idx], logger)
    X_val   = _apply_imputer(imputer, X_np[val_idx], logger)
    X_test  = _apply_imputer(imputer, X_np[test_idx], logger)

    y_train = y_np[train_idx]
    y_val   = y_np[val_idx]
    y_test  = y_np[test_idx]

    sex_train = sex_np[train_idx]
    sex_val   = sex_np[val_idx]
    sex_test  = sex_np[test_idx]

    strat_train = strat_labels[train_idx]
    strat_val   = strat_labels[val_idx]
    strat_test  = strat_labels[test_idx]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------
    timestamp_fmt = run_cfg["timestamp_format"]
    timestamp = datetime.now().strftime(timestamp_fmt)

    outdir = os.path.join(out_root, dataset_name, model_folder, f"preprocess_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"Saving preprocessing artifacts to: {outdir}")

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------
    _save_splits(train_idx, val_idx, test_idx, outdir)
    _save_feature_names(feature_cols, outdir)
    _save_imputer(imputer, outdir)

    # Metadata
    metadata = {
        "preprocess_version": "1.0",
        "preprocess_timestamp": timestamp,
        "dataset_name": dataset_name,
        "feature_file": feature_file,
        "outcome_file": outcome_file,
        "n_samples": len(X_np),
        "n_features": len(feature_cols),
        "study_id_dtype": "Int64",
        "majority_sex": majority,
        "sex_encoding_map": {"male": 1, "female": 0},
        "imputer_strategy": imputer_strategy,
        "random_seed": seed,
        "split": {
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "test_count": len(test_idx),
        },
    }

    _save_preprocess_metadata(metadata, outdir)

    logger.info("Preprocessing complete.")

    # --------------------------------------------------------
    # Return everything needed by training
    # --------------------------------------------------------
    return {
        "X_train": X_train,
        "y_train": y_train,
        "sex_train": sex_train,
        "strat_train": strat_train,

        "X_val": X_val,
        "y_val": y_val,
        "sex_val": sex_val,
        "strat_val": strat_val,

        "X_test": X_test,
        "y_test": y_test,
        "sex_test": sex_test,
        "strat_test": strat_test,

        "feature_names": feature_cols,
        "imputer": imputer,
        "outdir": outdir,
    }
