# loader.py

import os
import json
import yaml
import glob
import joblib
import polars as pl
from datetime import datetime


# ============================================================
# CONFIG LOADING
# ============================================================

def load_yaml_config(path: str) -> dict:
    """
    Load a YAML configuration file into a Python dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)



# ============================================================
# DATASET FILE LOADING
# ============================================================

def resolve_dataset_paths(dataset_name: str, run_cfg: dict) -> dict:
    """
    Build absolute paths for feature/outcome CSVs for a dataset.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in run_cfg["datasets"]:
        raise ValueError(
            f"Dataset '{dataset_name}' not allowed. Must be one of {run_cfg['datasets']}"
        )

    root = run_cfg["paths"]["input_root"]
    file_cfg = run_cfg["input_files"][dataset_name]

    features_path = os.path.join(root, file_cfg["features"])
    outcomes_path = os.path.join(root, file_cfg["outcomes"])

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features file: {features_path}")

    if not os.path.exists(outcomes_path):
        raise FileNotFoundError(f"Missing outcomes file: {outcomes_path}")

    return {
        "features": features_path,
        "outcomes": outcomes_path,
    }


def load_dataset_files(dataset_name: str, run_cfg: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load features and outcomes into Polars DataFrames.
    Renames StudyID → Study_ID in outcomes.
    """
    paths = resolve_dataset_paths(dataset_name, run_cfg)

    # Load features
    X = pl.read_csv(paths["features"])

    # Load outcomes
    Y = pl.read_csv(paths["outcomes"])
    if "StudyID" in Y.columns:
        Y = Y.rename({"StudyID": "Study_ID"})

    if "POMC" not in Y.columns:
        raise ValueError(
            f"Outcomes file does not contain required 'POMC' column: {paths['outcomes']}"
        )

    return X, Y



# ============================================================
# OUTPUT FOLDER PREPARATION (timestamped run folders)
# ============================================================

def prepare_output_paths(dataset_name: str, run_cfg: dict) -> str:
    """
    Create timestamped output folder for a dataset, following format:
        outputs/<dataset>/gbc/<timestamp>/
    Returns the path to the created run folder.
    """
    dataset_name = dataset_name.lower()
    base_out = run_cfg["paths"]["output_root"]
    model_folder = run_cfg["folder_naming"]["model_folder"]

    timestamp = datetime.now().strftime(run_cfg["timestamp_format"])
    out_dir = os.path.join(base_out, dataset_name, model_folder, timestamp)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir



# ============================================================
# METADATA SAVE/LOAD
# ============================================================

def save_metadata(obj: dict, path: str):
    """
    Save JSON metadata (splits, preprocessing config, results, etc).
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_metadata(path: str) -> dict:
    """
    Load metadata JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata JSON does not exist: {path}")

    with open(path, "r") as f:
        return json.load(f)



# ============================================================
# MODEL & ARTIFACT LOADING
# ============================================================

def load_model_artifacts(run_path: str) -> dict:
    """
    Load all artifacts from a run folder:
      - model.joblib
      - imputer.joblib (if exists)
      - feature_names.json
      - splits.json
      - training_results.json
      - evaluation_results.json (if exists)

    Returns a dict of available artifacts.
    """
    artifacts = {}

    model_path = os.path.join(run_path, "model.joblib")
    if os.path.exists(model_path):
        artifacts["model"] = joblib.load(model_path)

    imputer_path = os.path.join(run_path, "imputer.joblib")
    if os.path.exists(imputer_path):
        artifacts["imputer"] = joblib.load(imputer_path)

    splits_path = os.path.join(run_path, "splits.json")
    if os.path.exists(splits_path):
        artifacts["splits"] = load_metadata(splits_path)

    feat_path = os.path.join(run_path, "feature_names.json")
    if os.path.exists(feat_path):
        artifacts["feature_names"] = load_metadata(feat_path)

    train_meta_path = os.path.join(run_path, "training_results.json")
    if os.path.exists(train_meta_path):
        artifacts["training_results"] = load_metadata(train_meta_path)

    eval_meta_path = os.path.join(run_path, "evaluation_results.json")
    if os.path.exists(eval_meta_path):
        artifacts["evaluation_results"] = load_metadata(eval_meta_path)

    return artifacts



# ============================================================
# FINDING MOST RECENT RUN (for comparison or analyze actions)
# ============================================================

def discover_latest_run(dataset_name: str, run_cfg: dict) -> str | None:
    """
    Find the most recent timestamped run folder for a dataset.
    Returns path or None.
    """
    dataset_name = dataset_name.lower()
    base_out = run_cfg["paths"]["output_root"]
    model_folder = run_cfg["folder_naming"]["model_folder"]

    pattern = os.path.join(base_out, dataset_name, model_folder, "*")
    candidates = glob.glob(pattern)

    if not candidates:
        return None

    # Timestamped folder names → sort by name
    candidates = sorted(candidates)
    return candidates[-1]   # newest
