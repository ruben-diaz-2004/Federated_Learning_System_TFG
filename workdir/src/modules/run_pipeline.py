"""
@author: Ruben Diaz Marrero
Grado en ingenieria informatica, Universidad de La Laguna
Trabajo de Fin de Grado -- Curso 2025/2026
======================
run_pipeline.py
===============
Adversarial attack pipeline over an already-trained federated model.

The script does NOT train a model. It expects a TrainingResult to
already exist in the DB (produced by new_experiment.py) and runs
FGSM, PGD and BIM attacks over the model stored in that result.

Flow:
  1. Resolve result_id:
       - If --result_id is given, use it directly.
       - Otherwise fall back to the latest TrainingResult in the DB.
  2. Retrieve model_path and data context from TrainingResult.
  3. Run FGSM, PGD and BIM attacks via adversarial_attacks.run_attack().
  4. Persist each AdversarialRun in the DB linked to the result_id.

Usage -- attack the latest trained model:
    python run_pipeline.py --data_dir /path/to/dataset

Usage -- attack a specific result:
    python run_pipeline.py --data_dir /path/to/dataset --result_id 7

Prerequisites:
  - new_experiment.py has been run and at least one TrainingResult exists.
  - MySQL running with the credentials in database_access.DB_CONFIG.
  - ART and PyTorch installed.
"""

import argparse
import sys

from adversarial_attacks import run_attack
from database_access import (
    get_db,
    get_result_info,
    get_latest_result_id,
    register_adversarial_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_result_id(requested_id: int | None) -> int:
    """
    Returns the result_id to attack.

    If requested_id is provided it is validated against the DB first.
    If not provided the latest TrainingResult is used as fallback.
    Raises ValueError (printed and sys.exit) on any DB miss.
    """
    if requested_id is not None:
        # Validate that it actually exists -- get_result_info raises if not.
        get_result_info(requested_id)
        return requested_id
    print("[BD] --result_id not provided, falling back to latest TrainingResult...")
    return get_latest_result_id()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial attack pipeline (FGSM + PGD + BIM) over a model "
            "already registered in the DB via new_experiment.py."
        )
    )
    # Required
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the dataset directory used to generate adversarial examples.",
    )
    # Optional result selector
    parser.add_argument(
        "--result_id", type=int, default=None,
        help=(
            "result_id of the TrainingResult to attack. "
            "If omitted, the latest result in the DB is used."
        ),
    )
    # Attack hyperparameters
    parser.add_argument("--epsilon",    type=float, default=0.1,
                        help="Maximum perturbation magnitude (L-inf).")
    parser.add_argument("--max_iter",   type=int,   default=10,
                        help="Maximum attack iterations (PGD / BIM).")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="Batch size for inference during attacks.")
    parser.add_argument("--n_samples",  type=int,   default=None,
                        help="Limit number of samples per attack (None = all).")
    args = parser.parse_args()

    # ── Step 1: resolve result_id ─────────────────────────────────────────
    try:
        result_id = resolve_result_id(args.result_id)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Step 2: retrieve model path from DB ───────────────────────────────
    try:
        result_info = get_result_info(result_id)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    model_path = result_info["model_path"]

    print("\n══════════════════════════════════════════════════════")
    print(f" Attacking result_id={result_id}")
    print(f"   model_path      : {model_path}")
    print(f"   best_val_bal_acc: {result_info['best_val_bal_acc']:.4f}")
    print(f"   test_bal_acc    : {result_info['test_bal_acc']:.4f}")
    print("══════════════════════════════════════════════════════")

    # ── Step 3: define attacks ────────────────────────────────────────────
    attacks = [
        {
            "attack_type":     "fgsm",
            "epsilon":         args.epsilon,
            "eps_step":        None,
            "max_iter":        args.max_iter,
            "num_random_init": 1,
        },
        {
            "attack_type":     "pgd",
            "epsilon":         args.epsilon,
            "eps_step":        args.epsilon / 4,
            "max_iter":        args.max_iter,
            "num_random_init": 1,
        },
        {
            "attack_type":     "bim",
            "epsilon":         args.epsilon,
            "eps_step":        args.epsilon / 4,
            "max_iter":        args.max_iter,
            "num_random_init": 1,
        },
    ]

    # ── Step 4: run and persist each attack ───────────────────────────────
    print("\n══════════════════════════════════════════════════════")
    print(" PHASE 2 -- Adversarial attacks (ART)")
    print("══════════════════════════════════════════════════════")

    for atk in attacks:
        atype = atk["attack_type"].upper()
        print(f"\n-- Attack {atype} -------------------------------------------")
        try:
            metrics = run_attack(
                data_dir        = args.data_dir,
                model_path      = model_path,
                attack_type     = atk["attack_type"],
                epsilon         = atk["epsilon"],
                eps_step        = atk["eps_step"],
                max_iter        = atk["max_iter"],
                num_random_init = atk["num_random_init"],
                batch_size      = args.batch_size,
                n_samples       = args.n_samples,
                save_adv        = None,
            )

            adv_id = register_adversarial_run(
                result_id       = result_id,
                attack_type     = atk["attack_type"],
                epsilon         = atk["epsilon"],
                eps_step        = atk["eps_step"],
                max_iter        = atk["max_iter"],
                num_random_init = atk["num_random_init"],
                n_samples       = args.n_samples,
                clean_bal_acc   = metrics["clean_bal_acc"],
                adv_bal_acc     = metrics["adv_bal_acc"],
                clean_precision = metrics["clean_precision"],
                adv_precision   = metrics["adv_precision"],
                clean_recall    = metrics["clean_recall"],
                adv_recall      = metrics["adv_recall"],
            )

            drop = metrics["clean_bal_acc"] - metrics["adv_bal_acc"]
            print(f"  [BD] AdversarialRun saved -> adv_id={adv_id}")
            print(f"       Clean Bal-Acc : {metrics['clean_bal_acc']:.4f}")
            print(f"       Adv   Bal-Acc : {metrics['adv_bal_acc']:.4f}  "
                  f"(drop={drop:.4f})")

        except Exception as exc:
            print(f"  [WARN] Attack {atype} failed and was skipped: {exc}")

    print("\n══════════════════════════════════════════════════════")
    print(" Pipeline completed successfully")
    print("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()