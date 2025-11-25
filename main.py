# main.py

import argparse
import os
import sys
from datetime import datetime

from pipeline.logger import init_logger, log_timing, install_global_exception_hook

from pipeline.loader import (
    load_yaml_config,
    discover_latest_run,
    load_model_artifacts
)
from pipeline.preprocess import (
    preprocess_dataset,
    load_existing_preprocess
)
from pipeline.training import train_model
from pipeline.evaluator import run_evaluation, load_evaluation_results
from pipeline.reporting import generate_report
from pipeline.compare import run_comparison


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
        help="Comma-separated run folders for comparison (for --action compare)"
    )

    return parser


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",")]
    action = args.action

    # --------------------------------------------------------
    # Initialize logger
    # --------------------------------------------------------
    logger = init_logger(action)
    install_global_exception_hook(logger)

    with log_timing(logger, action.upper()):
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
                preprocess_dataset(ds, run_cfg, logger=logger)
            return

        # ========================================================
        # ACTIONS THAT REQUIRE PREPROCESS
        # ========================================================
        # (train / evaluate / report all need preprocess)
        preprocess_pkgs = {}

        for ds in datasets:
            print(f"\n=== PREPARING {ds.upper()} ===")

            if args.preprocess_folder:
                # User explicitly gave preprocess_<timestamp>
                preprocess_pkgs[ds] = load_existing_preprocess(
                    ds, run_cfg, logger=logger, 
                    # the user's load_existing_preprocess signature is (dataset_name, run_cfg, logger)
                    # so preprocess_folder is not accepted → manual folder override necessary
                )
                # override folder manually:
                preprocess_pkgs[ds]["preprocess_dir"] = os.path.join(
                    run_cfg["paths"]["output_root"],
                    ds,
                    run_cfg["folder_naming"]["model_folder"],
                    args.preprocess_folder
                )
            else:
                # autodetect latest preprocess
                preprocess_dir = discover_latest_run(ds, run_cfg)
                if preprocess_dir is None:
                    sys.exit(
                        f"ERROR: No preprocess_<timestamp> folder found for dataset {ds}. "
                        "Run --action preprocess first or specify --preprocess-folder."
                    )
                # load existing preprocess
                preprocess_pkgs[ds] = load_existing_preprocess(
                    ds, run_cfg, logger=logger
                )

        # ========================================================
        # ACTION: TRAIN
        # ========================================================
        if action == "train":
            for ds in datasets:
                print(f"\n=== TRAINING {ds.upper()} ===")

                preprocess_pkg = preprocess_pkgs[ds]

                train_model(
                    dataset_name=ds,
                    run_cfg=run_cfg,
                    model_cfg=model_cfg["models"]["gbc"],   # only gbc
                    preprocess_pkg=preprocess_pkg,
                    logger=logger
                )
            return

        # ========================================================
        # ACTION: EVALUATE
        # ========================================================
        if action == "evaluate":
            for ds in datasets:
                print(f"\n=== EVALUATING {ds.upper()} ===")

                # Determine run folder
                if args.run_folder:
                    run_folder = args.run_folder
                else:
                    run_folder = discover_latest_run(ds, run_cfg)
                    if run_folder is None or "preprocess_" in run_folder:
                        sys.exit(f"No run_<timestamp> folder found for dataset {ds}. Train first.")

                # Load all artifacts inside that run folder
                run_path = os.path.join(run_cfg["paths"]["output_root"], ds, run_folder)
                artifacts = load_model_artifacts(run_path)

                # Build model_results for evaluator
                y_test = artifacts.get("y_test")
                y_pred_test = artifacts.get("y_pred_test")
                sex_test = artifacts["training_results"].get("sex_test") if "training_results" in artifacts else None

                model_results = {
                    "y_test": y_test,
                    "y_pred_test": y_pred_test,
                    "y_pred_test_bin": (y_pred_test >= 0.5).astype(int),
                    "sex_test": sex_test,
                }

                eval_results = run_evaluation(ds, run_cfg, model_results, logger=logger)

                # Save eval results
                out_dir = run_path
                from pipeline.evaluator import save_evaluation_results
                save_evaluation_results(eval_results, out_dir)

                print(f"Evaluation complete for {ds}")

            return

        # ========================================================
        # ACTION: REPORT
        # ========================================================
        if action == "report":
            for ds in datasets:
                print(f"\n=== REPORT {ds.upper()} ===")

                # Determine run folder
                if args.run_folder:
                    run_folder = args.run_folder
                else:
                    run_folder = discover_latest_run(ds, run_cfg)
                    if run_folder is None or "preprocess_" in run_folder:
                        sys.exit(f"No run_<timestamp> folder for dataset {ds}. Train first.")

                run_path = os.path.join(run_cfg["paths"]["output_root"], ds, run_folder)

                # Load model artifacts
                artifacts = load_model_artifacts(run_path)
                training_metadata = artifacts["training_results"]

                # Load evaluation.json
                eval_results = load_evaluation_results(run_path)

                # Load preprocess metadata
                preprocess_folder = training_metadata["preprocess_folder"]
                preprocess_pkg = load_existing_preprocess(ds, run_cfg, logger=logger)
                preprocess_meta = preprocess_pkg["metadata"]

                generate_report(
                    dataset_name=ds,
                    run_cfg=run_cfg,
                    eval_results=eval_results,
                    training_metadata=training_metadata,
                    preprocess_meta=preprocess_meta,
                    run_folder=run_folder,
                    logger=logger
                )

                print(f"Report generated for {ds}")

            return

        # ========================================================
        # ACTION: COMPARE
        # ========================================================
        if action == "compare":
            print("\n=== COMPARISON MODE ===")

            if not args.runs:
                sys.exit("ERROR: --runs must be provided for comparison.")

            run_list = [r.strip() for r in args.runs.split(",")]

            # NOTE: your compare.py signature is:
            #   run_comparison(dataset_name_arg, run_cfg, compare_targets, logger)
            run_comparison(
                dataset_name_arg=",".join(datasets),
                run_cfg=run_cfg,
                compare_targets=run_list,
                logger=logger
            )

            print("Comparison complete.")
            return


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()
