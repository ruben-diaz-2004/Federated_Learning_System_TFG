"""
@author: Ruben Diaz Marrero
Grado en ingenieria informatica, Universidad de La Laguna
Trabajo de Fin de Grado -- Curso 2025/2026
======================
run_backdoor_pipeline.py
========================
Backdoor attack pipeline: poisoned federated training + Activation Clustering
defense + persistence in the DB.

Unlike run_pipeline.py and run_mia_pipeline.py, this script DOES train a model
because a backdoor attack requires retraining on poisoned data. The resulting
poisoned TrainingResult is registered fresh in the DB.

Flow:
  1. Retrieve dataset_id from DB (Dataset must already exist).
  2. Run poisoned federated training via backdoor_syft.run_backdoor_syft_pipeline()
     and register Split, Experiment, TrainingResult and ExperimentSplit in the DB.
  3. Run Activation Clustering defense via backdoor_defense.run_defense().
  4. Persist one PoisoningRun row combining attack + defense metrics,
     linked to the poisoned model's own result_id.

Usage:
    python run_backdoor_pipeline.py \\
        --data_dir       /path/to/dataset \\
        --dataset_name   RIMONE \\
        --model_out      backdoor_syft.pth \\
        --trigger_type   square \\
        --percent_poison 0.2

Prerequisites:
  - The Dataset row already exists in the DB (run create_dataset.py first).
  - MySQL running with the credentials in database_access.DB_CONFIG.
  - PySyft, ART and PyTorch installed.
"""

import argparse
import sys

from backdoor_syft    import run_backdoor_syft_pipeline
from backdoor_defense import run_defense
from database_access import (
    register_server,
    register_split_server,
    register_experiment,
    register_experiment_split,
    register_training_result,
    link_result_to_es,
    register_poisoning_run,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Backdoor pipeline: poisoned federated training + "
            "Activation Clustering defense + DB persistence."
        ),
    )
    # Dataset
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--dataset_name", required=True,
        help="Name of the dataset already registered in the Dataset table.",
    )
    parser.add_argument(
        "--model_out", default="backdoor_syft.pth",
        help="Output .pth file for the poisoned model.",
    )

    # Training hyperparameters
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=500)
    parser.add_argument("--patience",    type=int,   default=50)
    parser.add_argument("--num_classes", type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.10)

    # Backdoor attack parameters
    parser.add_argument(
        "--trigger_type", type=str, default="square",
        choices=["square", "cross", "checkerboard",
                 "gaussian", "sinusoidal", "border"],
    )
    parser.add_argument("--percent_poison",   type=float, default=0.2)
    parser.add_argument("--trigger_size",     type=int,   default=8)
    parser.add_argument(
        "--trigger_position", type=str, default="top_left",
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
    )
    parser.add_argument("--source_class", type=int, default=1)
    parser.add_argument("--target_class", type=int, default=0)

    # Defense parameters
    parser.add_argument(
        "--check_all_classes", action="store_true",
        help="Run Activation Clustering over all classes instead of target only.",
    )

    args = parser.parse_args()

    # ======================================================================
    # PHASE 1 -- Poisoned federated training
    # ======================================================================
    print("\n======================================================")
    print(" PHASE 1 -- Poisoned federated training")
    print("======================================================")
    print(f"  Trigger     : {args.trigger_type} "
          f"(size={args.trigger_size}, pos={args.trigger_position})")
    print(f"  Poison rate : {args.percent_poison}")
    print(f"  Source->tgt : {args.source_class} -> {args.target_class}")

    training_result = run_backdoor_syft_pipeline(
        data_dir         = args.data_dir,
        model_out        = args.model_out,
        cfg = {
            "batch_size":  args.batch_size,
            "epochs":      args.epochs,
            "lr":          args.lr,
            "patience":    args.patience,
            "num_classes": args.num_classes,
            "seed":        args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio":   args.val_ratio,
        },
        trigger_type     = args.trigger_type,
        percent_poison   = args.percent_poison,
        trigger_size     = args.trigger_size,
        trigger_position = args.trigger_position,
        source_class     = args.source_class,
        target_class     = args.target_class,
    )

    if training_result.get("status") != "Exito":
        print("\n[ERROR] Poisoned training failed:")
        print(training_result.get("error"))
        sys.exit(1)

    # -- Register poisoned model in DB ------------------------------------
    print("\n[BD] Registering poisoned training results...")

    # Ensure a generic local server row exists for non-federated pipelines.
    # register_server is idempotent: safe to call every run.
    LOCAL_SERVER_NAME = "local"
    register_server(
        name           = LOCAL_SERVER_NAME,
        owner_email    = "local@codigla.org",
        owner_password = "changethis",
    )

    split_id = register_split_server(
        dataset_name = args.dataset_name,
        server_name  = LOCAL_SERVER_NAME,
        model_path   = args.model_out,
        n_train      = training_result["train_samples"],
        n_val        = training_result["val_samples"],
        n_test       = training_result["test_samples"],
        seed         = args.seed,
        train_ratio  = args.train_ratio,
        val_ratio    = args.val_ratio,
    )

    experiment_id = register_experiment(
        eve_model_path = args.model_out,
        lr             = args.lr,
        batch_size     = args.batch_size,
        epochs_max     = args.epochs,
        patience       = args.patience,
        description    = (
            f"ResNet50 backdoor | dataset={args.dataset_name} | "
            f"trigger={args.trigger_type} | poison={args.percent_poison}"
        ),
    )

    es_id = register_experiment_split(experiment_id, split_id)

    result_id = register_training_result(
        model_path       = training_result["model_path"],
        best_epoch       = training_result["best_epoch"],
        best_val_bal_acc = training_result["best_val_bal_acc"],
        test_bal_acc     = training_result["clean_bal_acc"],
        test_precision   = training_result["clean_precision"],
        test_recall      = training_result["clean_recall"],
    )

    link_result_to_es(es_id, result_id)

    print(f"     split_id      = {split_id}")
    print(f"     experiment_id = {experiment_id}")
    print(f"     es_id         = {es_id}")
    print(f"     result_id     = {result_id}")
    print(f"     ASR           = {training_result['attack_success_rate']:.4f}")
    print(f"     Clean bal-acc = {training_result['clean_bal_acc']:.4f}")

    # ======================================================================
    # PHASE 2 -- Activation Clustering defense
    # ======================================================================
    print("\n======================================================")
    print(" PHASE 2 -- Activation Clustering defense")
    print("======================================================")

    classes_to_check = None if args.check_all_classes else [args.target_class]

    defense_result = run_defense(
        data_dir         = args.data_dir,
        model_path       = training_result["model_path"],
        trigger_type     = args.trigger_type,
        poison_rate      = args.percent_poison,
        trigger_size     = args.trigger_size,
        trigger_pos      = args.trigger_position,
        source_class     = args.source_class,
        target_class     = args.target_class,
        batch_size       = args.batch_size,
        classes_to_check = classes_to_check,
    )

    print(f"     AC precision  = {defense_result['precision']:.4f}")
    print(f"     AC recall     = {defense_result['recall']:.4f}")
    print(f"     AC F1         = {defense_result['f1']:.4f}")

    # ======================================================================
    # PHASE 3 -- Persist PoisoningRun
    # ======================================================================
    print("\n======================================================")
    print(" PHASE 3 -- Persisting PoisoningRun")
    print("======================================================")

    poison_id = register_poisoning_run(
        result_id           = result_id,
        trigger_type        = training_result["trigger_type"],
        trigger_size        = training_result["trigger_size"],
        trigger_position    = training_result["trigger_position"],
        percent_poison      = training_result["percent_poison"],
        n_poisoned          = training_result["n_poisoned"],
        source_class        = training_result["source_class"],
        target_class        = training_result["target_class"],
        clean_bal_acc       = training_result["clean_bal_acc"],
        clean_precision     = training_result["clean_precision"],
        clean_recall        = training_result["clean_recall"],
        attack_success_rate = training_result["attack_success_rate"],
        ac_precision        = defense_result["precision"],
        ac_recall           = defense_result["recall"],
        ac_f1               = defense_result["f1"],
    )

    print(f"     poison_id     = {poison_id}")

    print("\n======================================================")
    print(" Backdoor pipeline completed successfully")
    print("======================================================")


if __name__ == "__main__":
    main()