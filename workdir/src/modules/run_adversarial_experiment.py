"""
run_adversarial_experiment.py
==============================
Adversarial attack pipeline driven entirely by the database.

Usage -- register a new experiment from a prior training experiment and run it:
    python run_adversarial_experiment.py \\
        --register \\
        --name                   adv_rimone_v1 \\
        --training_experiment_id 95 \\
        --data_dir               /path/to/rimone_x

Usage -- re-run an already-registered experiment:
    python run_adversarial_experiment.py \\
        --name     adv_rimone_v1 \\
        --data_dir /path/to/rimone_x

Flow
----
  1. Look up the Experiment row by name (or register it if --register is given).
  2. When registering, copy splits+result_ids from the training experiment so
     every ExperimentSplit already has a result_id pointing to the trained model.
  3. Read AdversarialExperimentParams for hyperparameters.
  4. For each split grouped by server:
       a. Retrieve result_id -> model_path from TrainingResult.
       b. Run each requested attack type against that model.
       c. Persist one AdversarialRun row per attack, linked to result_id.
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

from adversarial_attacks import run_attack
from database_access import (
    register_attack_experiment,
    register_adversarial_experiment_params,
    get_experiment_by_name,
    get_experiment_splits,
    get_adversarial_experiment_params,
    get_result_info,
    get_server,
    register_adversarial_run,
    get_db,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_splits_from_training(adv_experiment_id: int,
                               training_experiment_id: int) -> int:
    """
    Copy ExperimentSplit rows from a training experiment into the new attack
    experiment, preserving the existing result_id so the attack script can
    immediately find the trained model for each split.

    Parameters
    ----------
    adv_experiment_id : int
        The newly created adversarial experiment.
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
        cur.execute(sql, (adv_experiment_id, training_experiment_id))
        return cur.rowcount


def build_attack_configs(params: dict) -> list:
    """
    Expand the attack_types string from AdversarialExperimentParams into a
    list of per-attack config dicts ready for run_attack().
    """
    epsilon      = params["epsilon"]
    eps_step_raw = params["eps_step"]
    max_iter     = params["max_iter"]
    n_rand       = params["num_random_init"]

    configs = []
    for atype in [a.strip() for a in params["attack_types"].split(",")]:
        # FGSM is single-step: eps_step and restarts are not used.
        step = None if atype == "fgsm" else (
            eps_step_raw if eps_step_raw is not None else epsilon / 4.0
        )
        configs.append({
            "attack_type":     atype,
            "epsilon":         epsilon,
            "eps_step":        step,
            "max_iter":        max_iter,
            "num_random_init": n_rand,
        })
    return configs


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_new_experiment(args) -> int:
    """
    Create Experiment + AdversarialExperimentParams, then copy splits from
    the training experiment (with their result_ids already filled in).

    Returns experiment_id.
    """
    exp_id = register_attack_experiment(
        name            = args.name,
        experiment_type = "adversarial",
        description     = args.description,
    )
    print(f"[DB] Experiment registered: experiment_id={exp_id}, name='{args.name}'")

    register_adversarial_experiment_params(
        experiment_id   = exp_id,
        attack_types    = args.attack_types,
        epsilon         = args.epsilon,
        eps_step        = args.eps_step,
        max_iter        = args.max_iter,
        num_random_init = args.num_random_init,
        n_samples       = args.n_samples,
        batch_size      = args.batch_size,
    )
    print("[DB] AdversarialExperimentParams saved.")

    n = copy_splits_from_training(exp_id, args.training_experiment_id)
    print(f"[DB] {n} split(s) copied from training experiment "
          f"{args.training_experiment_id} (result_ids preserved).")

    return exp_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial attack pipeline driven by the DB experiment registry."
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
    parser.add_argument("--description",     default=None)
    parser.add_argument("--attack_types",    default="fgsm,pgd,bim",
                        help="Comma-separated attack types.")
    parser.add_argument("--epsilon",         type=float, default=0.1)
    parser.add_argument("--eps_step",        type=float, default=None,
                        help="Step size for PGD/BIM; default is epsilon/4.")
    parser.add_argument("--max_iter",        type=int,   default=10)
    parser.add_argument("--num_random_init", type=int,   default=1)
    parser.add_argument("--n_samples",       type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=32)

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

    # -- Step 2: load attack hyperparameters ---------------------------------
    params         = get_adversarial_experiment_params(exp_id)
    attack_configs = build_attack_configs(params)
    print(f"[DB] Attack types : {params['attack_types']}")
    print(f"     epsilon={params['epsilon']}  max_iter={params['max_iter']}")

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

            # -- Step 4: run each attack and persist results -----------------
            for atk in attack_configs:
                atype = atk["attack_type"].upper()
                print(f"\n  -- Attack {atype} --")
                try:
                    metrics = run_attack(
                        data_dir        = args.data_dir,
                        model_path      = model_path,
                        attack_type     = atk["attack_type"],
                        epsilon         = atk["epsilon"],
                        eps_step        = atk["eps_step"],
                        max_iter        = atk["max_iter"],
                        num_random_init = atk["num_random_init"],
                        batch_size      = params["batch_size"],
                        n_samples       = params["n_samples"],
                        save_adv        = None,
                    )

                    adv_id = register_adversarial_run(
                        result_id       = result_id,
                        attack_type     = atk["attack_type"],
                        epsilon         = atk["epsilon"],
                        eps_step        = atk["eps_step"],
                        max_iter        = atk["max_iter"],
                        num_random_init = atk["num_random_init"],
                        n_samples       = params["n_samples"],
                        clean_bal_acc   = metrics["clean_bal_acc"],
                        adv_bal_acc     = metrics["adv_bal_acc"],
                        clean_precision = metrics["clean_precision"],
                        adv_precision   = metrics["adv_precision"],
                        clean_recall    = metrics["clean_recall"],
                        adv_recall      = metrics["adv_recall"],
                    )

                    drop = metrics["clean_bal_acc"] - metrics["adv_bal_acc"]
                    print(f"     [DB] AdversarialRun saved -> adv_id={adv_id}")
                    print(f"          Clean Bal-Acc : {metrics['clean_bal_acc']:.4f}")
                    print(f"          Adv   Bal-Acc : {metrics['adv_bal_acc']:.4f}  "
                          f"(drop={drop:.4f})")

                except Exception as exc:
                    print(f"  [WARN] Attack {atype} on split_id={split_id} "
                          f"failed and was skipped: {exc}")

    print(f"\n{'=' * 60}")
    print(" Adversarial experiment completed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()