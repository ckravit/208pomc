# main.py

import argparse
import os
import sys
from datetime import datetime

from loader import (
    load_yaml_config,
    discover_latest_run,
    load_model_artifacts
)
from preprocess import (
    preprocess_dataset,
    load_existing_preprocess
)
from training import train_model
from evaluator import run_evaluation
from reporting import generate_report
from compare import run_comparison


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline Orchestrator for ABP/PPG ML Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["preprocess", "train", "evaluate", "report", "compare"],
        help="Pipeline action to perform."
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="abp",
        help="Comma-separated list: abp, ppg"
    )

    parser.add_argument(
        "--preprocess-folder",
        type=str,
        default=None,
        help="Optional: specify preprocess_<timestamp> folder"
    )

    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help="Optional: specify run_<timestamp> folder"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gbc",
        help="Model name (currently only gbc)"
    )

    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated run folders for comparison"
    )

    return parser


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    # --------------------------------------------------------
    # Parse CLI
    # --------------------------------------------------------
    parser = build_arg_parser()
    args = parser.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",")]
    action = args.action

    # --------------------------------------------------------
    # Load configs
    # --------------------------------------------------------
    run_cfg = load_yaml_config("run_config.yaml")
    model_cfg = load_yaml_config("model_config.yaml")

    # --------------------------------------------------------
    # ACTION: PREPROCESS
    # --------------------------------------------------------
    if action == "preprocess":
        for ds in datasets:
            print(f"\n=== PREPROCESSING {ds.upper()} ===")
            preprocess_dataset(ds, run_cfg, logger=None)  # logger optional
        return

    # --------------------------------------------------------
    # ACTIONS THAT REQUIRE PREPROCESS
    # --------------------------------------------------------
    for ds in datasets:
        print(f"\n=== PREPARING {ds.upper()} ===")

        # If user supplied preprocess folder → use it
        if args.preprocess_folder:
            preprocess_pkg = load_existing_preprocess(
                ds,
                run_cfg,
                preprocess_folder=args.preprocess_folder,
                logger=None
            )
        else:
            # Autodetect latest
            preprocess_pkg = discover_latest_run(
                base_path=os.path.join(run_cfg["paths"]["output_root"], ds),
                prefix="preprocess_"
            )
            if preprocess_pkg is None:
                sys.exit(
                    f"ERROR: No preprocess folder found for dataset {ds}. "
                    f"Run --action preprocess first or specify --preprocess-folder."
                )

    # --------------------------------------------------------
    # ACTION: TRAIN
    # --------------------------------------------------------
    if action == "train":
        for ds in datasets:
            print(f"\n=== TRAINING {ds.upper()} ===")

            # Load preprocess
            if args.preprocess_folder:
                preprocess_pkg = load_existing_preprocess(
                    ds, run_cfg, args.preprocess_folder, logger=None
                )
            else:
                preprocess_pkg = load_existing_preprocess(
                    ds, run_cfg, preprocess_pkg, logger=None
                )

            train_model(
                dataset_name=ds,
                run_cfg=run_cfg,
                model_cfg=model_cfg["gbc"],  # only gbc for now
                preprocess_pkg=preprocess_pkg,
                logger=None,
            )
        return

    # --------------------------------------------------------
    # ACTION: EVALUATE
    # --------------------------------------------------------
    if action == "evaluate":
        for ds in datasets:
            print(f"\n=== EVALUATING {ds.upper()} ===")

            if args.run_folder:
                run_folder = args.run_folder
            else:
                run_folder = discover_latest_run(
                    base_path=os.path.join(run_cfg["paths"]["output_root"], ds),
                    prefix="run_"
                )
                if run_folder is None:
                    sys.exit(f"No run folder available. Train first.")

            training_metadata, model, y_test, pred_test, sex_test = \
                load_model_artifacts(ds, run_folder, run_cfg)

            model_results = {
                "y_test": y_test,
                "y_test_prob": pred_test,
                "y_test_pred_bin": (pred_test >= 0.5).astype(int),
                "sex_test": sex_test,
            }

            eval_results = run_evaluation(
                ds, run_cfg, model_results, logger=None
            )

            print(f"Evaluation complete for {ds}")

        return

    # --------------------------------------------------------
    # ACTION: REPORT
    # --------------------------------------------------------
    if action == "report":
        for ds in datasets:
            print(f"\n=== REPORT {ds.upper()} ===")

            # Find run folder
            if args.run_folder:
                run_folder = args.run_folder
            else:
                run_folder = discover_latest_run(
                    base_path=os.path.join(run_cfg["paths"]["output_root"], ds),
                    prefix="run_"
                )

            training_metadata, model, y_test, pred_test, sex_test = \
                load_model_artifacts(ds, run_folder, run_cfg)

            # Load preprocess metadata
            preprocess_folder = training_metadata["preprocess_folder"]
            preprocess_pkg = load_existing_preprocess(
                ds, run_cfg, preprocess_folder, logger=None
            )
            preprocess_meta = preprocess_pkg["metadata"]

            eval_results = load_evaluation_results(
                os.path.join(
                    run_cfg["paths"]["output_root"],
                    ds,
                    run_folder
                )
            )

            from reporting import generate_report
            generate_report(
                dataset_name=ds,
                run_cfg=run_cfg,
                eval_results=eval_results,
                training_metadata=training_metadata,
                preprocess_meta=preprocess_meta,
                run_folder=run_folder,
                logger=None
            )

            print(f"Report generated for {ds}")
        return

    # --------------------------------------------------------
    # ACTION: COMPARE
    # --------------------------------------------------------
    if action == "compare":
        print("\n=== COMPARISON MODE ===")

        if not args.runs:
            sys.exit("ERROR: --runs must be provided for comparison.")

        run_list = [r.strip() for r in args.runs.split(",")]
        run_comparison(datasets, run_list, run_cfg)

        print("Comparison complete.")
        return


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()
