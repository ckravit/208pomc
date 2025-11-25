
# 208POMC — Waveform-Based POMC Classification Pipeline

This repository contains a fully modular machine learning pipeline for training, evaluating, reporting, and comparing predictive models for **Postoperative Major Complications (POMC)** using **ABP** and **PPG** waveform–derived features.

The pipeline supports:

- Modular preprocessing  
- Config‑driven model training  
- Evaluation (ROC/PR/AP/Brier/confusion)  
- Sex‑stratified subgroup performance  
- Markdown report generation  
- Automatic plot generation  
- Multi‑run comparison (DeLong + McNemar)  
- Timestamped, reproducible folder outputs  
- Centralized Python logging (per‑action log files)

All pipeline components live in `pipeline/` and are orchestrated through:

```
python main.py
```

## Models implemented
For now, only the Gradient Boost Classifier (GBC) has been implemented.

---

## 🔎 Quick Start

```bash
# 1. Preprocess both datasets
python main.py --action preprocess --datasets abp,ppg

# 2. Train model (uses latest preprocess)
python main.py --action train --datasets abp

# 3. Evaluate trained model
python main.py --action evaluate --datasets abp

# 4. Generate report + plots
python main.py --action report --datasets abp

# 5. Compare two runs
python main.py --action compare --datasets abp   --runs run_20250117_210055,run_20250118_093212
```

---

# 🚀 Full Usage Guide

## General CLI pattern
```
python main.py --action <ACTION> --datasets <abp,ppg> [other options]
```

If omitted, `--datasets` defaults to **abp**.

---

# 1️⃣ PREPROCESSING

Creates a new timestamped `preprocess_<timestamp>` folder:

```
python main.py --action preprocess --datasets abp,ppg
```

Artifacts saved under:
```
outputs/<dataset>/gbc/preprocess_<timestamp>/
    splits.json
    feature_names.json
    imputer.joblib
    metadata.json
```

---

# 2️⃣ TRAINING

Train a model using an existing preprocess folder.

Use latest automatically:
```
python main.py --action train --datasets abp
```

Explicit preprocess folder:
```
python main.py --action train --datasets ppg   --preprocess-folder preprocess_20250118_142233
```

Artifacts saved:
```
outputs/<dataset>/gbc/run_<timestamp>/
    model.joblib
    preds_test.npy
    y_test.npy
    training_metadata.json
```

---

# 3️⃣ EVALUATION

Runs all evaluation metrics and saves `evaluation.json`.

Evaluate latest run:
```
python main.py --action evaluate --datasets abp
```

Specify a run folder:
```
python main.py --action evaluate   --datasets ppg   --run-folder run_20250118_145012
```

Evaluation output:
```
outputs/<dataset>/gbc/run_<timestamp>/evaluation.json
```

---

# 4️⃣ REPORT GENERATION

Creates markdown summary of all metrics + plots:

```
python main.py --action report --datasets abp
```

Outputs:
```
outputs/<dataset>/gbc/run_<timestamp>/report/report.md
outputs/<dataset>/gbc/run_<timestamp>/plots/*.png
```

Plots include:
- ROC curve  
- PR curve  
- Calibration curve  
- Probability histogram  
- Sex‑stratified ROC/PR  
- Feature importances  
- SHAP (if enabled)

---

# 5️⃣ MODEL COMPARISON

Compare two or more runs using statistical tests.

```
python main.py --action compare --datasets abp   --runs run_20250117_210055,run_20250118_093212
```

Outputs:
```
outputs/compare/<dataset>/compare_<timestamp>.md
```
Includes:
- AUROC difference (DeLong)
- Classification disagreement (McNemar)
- Per-run metrics summary

---

# 📁 Project Structure

```
208pomc/
│
├── main.py                 # CLI entrypoint
├── run_config.yaml         # paths, seeds, preprocessing settings
├── model_config.yaml       # model hyperparameter grids
│
├── pipeline/
│   ├── loader.py           # loading YAML, artifacts, datasets
│   ├── logger.py           # log configuration
│   ├── preprocess.py       # preprocessing + stratified splits
│   ├── training.py         # training + saving metadata
│   ├── evaluator.py        # metric computation
│   ├── plotting.py         # all plots (ROC, PR, calibration, SHAP)
│   ├── reporting.py        # markdown model report generator
│   └── compare.py          # multi-run statistical comparison
│
├── inputs/
├── outputs/
└── logs/
```

Output hierarchy:
```
outputs/<dataset>/gbc/
    preprocess_<timestamp>/
    run_<timestamp>/
```

---

# 🧪 End‑to‑End Example

```
python main.py --action preprocess --datasets abp,ppg
python main.py --action train --datasets abp
python main.py --action evaluate --datasets abp
python main.py --action report --datasets abp
python main.py --action compare --datasets abp   --runs run_1,run_2
```

---

# 📊 Visual Diagram

```
     raw CSVs
         │
         ▼
──────────────────────────
     PREPROCESSING
  stratified splits
  imputation
  metadata/features
──────────────────────────
         │
         ▼
──────────────────────────
        TRAINING
  hyperparameter search
  final model fit
  predictions saved
──────────────────────────
         │
         ▼
──────────────────────────
        EVALUATE
  ROC/PR/AP/Brier
  confusion metrics
  sex‑stratified metrics
──────────────────────────
         │
         ▼
──────────────────────────
   REPORT + PLOTTING
  markdown report
  ROC/PR/calibration
  SHAP + feature import
──────────────────────────
         │
         ▼
──────────────────────────
        COMPARE
  statistical tests
  comparison markdown
──────────────────────────
```

---

# 📝 Logging

All pipeline steps automatically create log files in `logs/`.

Log naming pattern:
```
logs/<timestamp>_<action>.log
```

Example:
```
logs/20250118_150233_train.log
```

Each log captures:
- Action start + end times  
- Per‑dataset step timing  
- Key configuration values  
- Errors / warnings  
- Console output redirected  
- Critical pipeline decisions (autodetected run folders, etc.)

---

📝 Configuration Files

The pipeline uses two YAML files to control behavior without modifying code:

```run_config.yaml```
Defines global pipeline settings, including:
- locations of input CSV files
- where outputs are written
- timestamp format
- preprocessing options (e.g., seed, imputer strategy)
- dataset-specific file names

This file governs how data flows through the pipeline and how output folders are created.

```model_config.yaml```
Defines model-specific settings, including:
- which model(s) are available
- hyperparameter grids used during training
- any model-level default parameters

This file allows you to adjust or add models without touching the training code.
