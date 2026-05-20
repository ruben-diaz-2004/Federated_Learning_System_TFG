"""
@author: Ruben Diaz Marrero
Grado en ingenieria informatica, Universidad de La Laguna
Trabajo de Fin de Grado -- Curso 2025/2026
======================
run_mia_pipeline.py
===================
Membership Inference Attack pipeline over an already-trained federated model.

The script does NOT train a model. It expects a TrainingResult to already
exist in the DB (produced by new_experiment.py) and runs one or all MIA
variants over the model stored in that result.

Flow:
  1. Resolve result_id:
       - If --result_id is given, use it directly.
       - Otherwise fall back to the latest TrainingResult in the DB.
  2. Retrieve model_path from TrainingResult (overridable with --model_path).
  3. Run the requested MIA variant(s) via membership_inference.
  4. Persist each MembershipInferenceRun in the DB linked to result_id.

Usage -- single variant, latest model:
    python run_mia_pipeline.py --data_dir /path/to/dataset --variant rf

Usage -- all variants, specific result:
    python run_mia_pipeline.py --data_dir /path/to/dataset --result_id 7 --variant all
"""

import argparse
import sys

from membership_inference import run_mia, run_all_variants, SUPPORTED_VARIANTS
from database_access import (
    get_result_info,
    get_latest_result_id,
    register_mia_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_result_id(requested_id: int | None) -> int:
    """
    Returns the result_id to attack.
    Validates existence via get_result_info if explicitly provided.
    Falls back to get_latest_result_id() when None.
    """
    if requested_id is not None:
        get_result_info(requested_id)   # raises ValueError if not found
        return requested_id
    print("[BD] --result_id not provided, falling back to latest TrainingResult...")
    return get_latest_result_id()


def save_mia_to_db(result_id: int, metrics: dict) -> int:
    """
    Persists a metrics dict (returned by run_mia) into MembershipInferenceRun.
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
    print(f"  [BD] MembershipInferenceRun saved -> mia_id={mia_id} "
          f"(variant={metrics['variant']}, accuracy={metrics['mia_accuracy']:.4f})")
    return mia_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MIA pipeline: Membership Inference Attack over a model "
            "already registered in the DB via new_experiment.py."
        )
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--result_id", type=int, default=None,
        help=(
            "result_id of the TrainingResult to attack. "
            "If omitted, the latest result in the DB is used."
        ),
    )
    parser.add_argument(
        "--model_path", default=None,
        help=(
            "Path to the .pth model file. Optional -- by default the path "
            "stored in TrainingResult is used. Provide this only if you need "
            "to attack a different checkpoint than the one registered in the DB."
        ),
    )
    parser.add_argument(
        "--variant", type=str, default="all",
        choices=list(SUPPORTED_VARIANTS) + ["all"],
        help="MIA variant to run (default: all). 'all' runs every supported variant.",
    )
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--n_train_max", type=int, default=None,
                        help="Limit number of train samples used (None = all).")
    parser.add_argument("--n_test_max",  type=int, default=None,
                        help="Limit number of test samples used (None = all).")
    args = parser.parse_args()

    # ── Step 1: resolve result_id ─────────────────────────────────────────
    try:
        result_id = resolve_result_id(args.result_id)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Step 2: retrieve model path ───────────────────────────────────────
    try:
        result_info = get_result_info(result_id)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    db_model_path = result_info["model_path"]
    print(f"\n[BD] result_id={result_id} | model_path in DB: {db_model_path}")

    if args.model_path is None:
        model_path = db_model_path
    else:
        model_path = args.model_path
        if model_path != db_model_path:
            print(f"  [WARN] --model_path ({model_path}) differs from DB path. "
                  "Using the one provided via argument.")

    print(f"     Using model: {model_path}")

    # ── Step 3 & 4: run attack(s) and persist ────────────────────────────
    print("\n======================================================")
    print(" Membership Inference Attack")
    print("======================================================")

    if args.variant == "all":
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
                print(f"  [SKIP] {variant}: attack failed, not saved to DB.")
                continue
            try:
                save_mia_to_db(result_id, metrics)
                saved += 1
            except Exception as exc:
                print(f"  [WARN] Could not save variant '{variant}' to DB: {exc}")

        print(f"\n  Variants saved: {saved}/{len(SUPPORTED_VARIANTS)}")

    else:
        metrics = run_mia(
            data_dir    = args.data_dir,
            model_path  = model_path,
            variant     = args.variant,
            batch_size  = args.batch_size,
            n_train_max = args.n_train_max,
            n_test_max  = args.n_test_max,
        )

        try:
            save_mia_to_db(result_id, metrics)
        except Exception as exc:
            print(f"  [ERROR] Could not save to DB: {exc}")
            sys.exit(1)

    print("\n======================================================")
    print(" MIA pipeline completed successfully")
    print("======================================================")


if __name__ == "__main__":
    main()