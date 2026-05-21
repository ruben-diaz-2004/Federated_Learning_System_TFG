"""
run_mia_experiment.py
=====================
Membership Inference Attack pipeline driven entirely by the database.

Usage -- register a new experiment from a prior training experiment and run it:
    python run_mia_experiment.py \\
        --register \\
        --name                   mia_rimone_v1 \\
        --training_experiment_id 95 \\
        --data_dir               /path/to/rimone_x

Usage -- re-run an already-registered experiment:
    python run_mia_experiment.py \\
        --name     mia_rimone_v1 \\
        --data_dir /path/to/rimone_x

Flow
----
  1. Look up the Experiment row by name (or register it if --register is given).
  2. When registering, copy splits+result_ids from the training experiment so
     every ExperimentSplit already has a result_id pointing to the trained model.
  3. Read MiaExperimentParams for hyperparameters.
  4. For each split grouped by server:
       a. Retrieve result_id -> model_path from TrainingResult.
       b. Run the requested MIA variant(s) against that model.
       c. Persist one MembershipInferenceRun row per variant, linked to result_id.
  5. Per-server output allows independent statistical analysis afterwards.

Prerequisites
-------------
  - migration_experiment_type.sql has been applied to the DB.
  - A training experiment with result_ids already exists.
  - MySQL running with credentials in database_access.DB_CONFIG.
  - ART and PyTorch installed.
"""

import argparse
import sys

from membership_inference import run_mia, run_all_variants, SUPPORTED_VARIANTS
from database_access import (
    register_attack_experiment,
    register_mia_experiment_params,
    get_experiment_by_name,
    get_mia_experiment_params,
    get_experiment_splits,
    get_result_info,
    get_server,
    register_mia_run,
    get_db,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_splits_from_training(attack_experiment_id: int,
                               training_experiment_id: int) -> int:
    """
    Copy ExperimentSplit rows from a training experiment into the new attack
    experiment, preserving the existing result_id so the attack script can
    immediately find the trained model for each split.

    Parameters
    ----------
    attack_experiment_id : int
        The newly created MIA experiment.
    training_experiment_id : int
        The source training experiment whose splits and result_ids are reused.

    Returns
    -------
    int
        Number of splits copied.
    """
    sql = """
        INSERT INTO ExperimentSplit (experiment_id, split_id, result_id)
        SELECT %s, split_id, result_id
        FROM ExperimentSplit
        WHERE experiment_id = %s
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (attack_experiment_id, training_experiment_id))
        return cur.rowcount


def save_mia_to_db(result_id: int, metrics: dict) -> int:
    """
    Persist a metrics dict returned by run_mia() into MembershipInferenceRun.
    Returns the generated mia_id.
    """
    mia_id = register_mia_run(
        result_id       = result_id,
        attack_variant  = metrics["variant"],
        n_train_samples = metrics["n_train_samples"],
        n_test_samples  = metrics["n_test_samples"],
        mia_accuracy    = metrics["mia_accuracy"],
        mia_precision   = metrics["mia_precision"],
        mia_recall      = metrics["mia_recall"],
    )
    print(f"     [DB] MembershipInferenceRun saved -> mia_id={mia_id} "
          f"(variant={metrics['variant']}, accuracy={metrics['mia_accuracy']:.4f})")
    return mia_id


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_new_experiment(args) -> int:
    """
    Create Experiment + MiaExperimentParams, then copy splits from the
    training experiment (with their result_ids already filled in).

    Returns experiment_id.
    """
    exp_id = register_attack_experiment(
        name            = args.name,
        experiment_type = "mia",
        description     = args.description,
    )
    print(f"[DB] Experiment registered: experiment_id={exp_id}, name='{args.name}'")

    register_mia_experiment_params(
        experiment_id    = exp_id,
        variants         = args.variants,
        test_size        = args.test_size,
        n_shadow_samples = args.n_shadow_samples,
    )
    print("[DB] MiaExperimentParams saved.")

    n = copy_splits_from_training(exp_id, args.training_experiment_id)
    print(f"[DB] {n} split(s) copied from training experiment "
          f"{args.training_experiment_id} (result_ids preserved).")

    return exp_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MIA pipeline driven by the DB experiment registry."
    )

    # Experiment identification
    parser.add_argument(
        "--name", required=True,
        help="Unique experiment name (used to look up or register the experiment).",
    )
    parser.add_argument(
        "--register", action="store_true",
        help="Register a new experiment before running it.",
    )

    # Data
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the dataset directory.",
    )

    # Registration args
    parser.add_argument(
        "--training_experiment_id", type=int, default=None,
        help="experiment_id of the training experiment whose splits are reused "
             "(required with --register).",
    )
    parser.add_argument("--description",      default=None)
    parser.add_argument("--variants",         default="all",
                        help="Comma-separated MIA variants or 'all'.")
    parser.add_argument("--test_size",        type=float, default=0.5,
                        help="Fraction of shadow data used as MIA test set.")
    parser.add_argument("--n_shadow_samples", type=int,   default=None,
                        help="Number of shadow samples; None means all available.")

    # Runtime args (used every run, not stored in DB)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--n_train_max", type=int, default=None,
                        help="Limit number of train samples (None = all).")
    parser.add_argument("--n_test_max",  type=int, default=None,
                        help="Limit number of test samples (None = all).")

    args = parser.parse_args()

    # -- Step 1: resolve experiment ------------------------------------------
    if args.register:
        if args.training_experiment_id is None:
            print("[ERROR] --training_experiment_id is required with --register.")
            sys.exit(1)
        exp_id = register_new_experiment(args)
    else:
        exp    = get_experiment_by_name(args.name)
        exp_id = exp["experiment_id"]
        print(f"[DB] Loaded experiment '{args.name}' -> experiment_id={exp_id}")

    # -- Step 2: load MIA hyperparameters ------------------------------------
    params   = get_mia_experiment_params(exp_id)
    variants = params["variants"]
    print(f"[DB] Variants : {variants}")
    print(f"     test_size={params['test_size']}")

    # Resolve variant list for run_mia / run_all_variants
    run_all = (variants.strip().lower() == "all")
    if not run_all:
        variant_list = [v.strip() for v in variants.split(",")]
    else:
        variant_list = list(SUPPORTED_VARIANTS)

    # -- Step 3: iterate over splits grouped by server -----------------------
    splits_by_server = get_experiment_splits(exp_id)

    if not splits_by_server:
        print("[WARN] No splits linked to this experiment. Exiting.")
        sys.exit(0)

    print(f"\n[INFO] Servers involved: {list(splits_by_server.keys())}")

    for server_id_str, split_list in splits_by_server.items():
        server_info = get_server(int(server_id_str))
        print(f"\n{'=' * 60}")
        print(f" Server: {server_info['name']}  (server_id={server_id_str})")
        print(f"{'=' * 60}")

        for split in split_list:
            split_id  = split["split_id"]
            result_id = split.get("result_id")

            if result_id is None:
                print(f"  [SKIP] split_id={split_id} has no result_id.")
                continue

            result_info = get_result_info(result_id)
            model_path  = result_info["model_path"]

            print(f"\n  split_id={split_id}  result_id={result_id}")
            print(f"  model_path={model_path}")
            print(f"  test_bal_acc (training) = {result_info['test_bal_acc']:.4f}")

            print(f"\n  -- Membership Inference Attack --")

            # -- Step 4: run variant(s) and persist results ------------------
            if run_all:
                results = run_all_variants(
                    data_dir    = args.data_dir,
                    model_path  = model_path,
                    batch_size  = args.batch_size,
                    n_train_max = args.n_train_max,
                    n_test_max  = args.n_test_max,
                )
                saved = 0
                for variant, metrics in results.items():
                    if "error" in metrics:
                        print(f"  [SKIP] variant={variant}: attack failed, not saved.")
                        continue
                    try:
                        save_mia_to_db(result_id, metrics)
                        saved += 1
                    except Exception as exc:
                        print(f"  [WARN] Could not save variant '{variant}': {exc}")
                print(f"  Variants saved: {saved}/{len(SUPPORTED_VARIANTS)}")

            else:
                for variant in variant_list:
                    try:
                        metrics = run_mia(
                            data_dir    = args.data_dir,
                            model_path  = model_path,
                            variant     = variant,
                            batch_size  = args.batch_size,
                            n_train_max = args.n_train_max,
                            n_test_max  = args.n_test_max,
                        )
                        save_mia_to_db(result_id, metrics)
                    except Exception as exc:
                        print(f"  [WARN] variant={variant} on split_id={split_id} "
                              f"failed and was skipped: {exc}")

    print(f"\n{'=' * 60}")
    print(" MIA experiment completed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
