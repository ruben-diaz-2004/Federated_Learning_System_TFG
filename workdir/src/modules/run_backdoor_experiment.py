"""
run_backdoor_experiment.py
==========================
Backdoor attack pipeline driven entirely by the database.

Unlike adversarial and MIA experiments, backdoor DOES train a new model on
poisoned data, so it does not reuse splits from a prior training experiment.
Instead, a fresh Split and TrainingResult are created for each run.

Usage -- register a new experiment and run it immediately:
    python run_backdoor_experiment.py \\
        --register \\
        --name           bd_rimone_square_v1 \\
        --dataset_name   rimone \\
        --data_dir       /path/to/rimone_x \\
        --trigger_type   square \\
        --percent_poison 0.2

Usage -- re-run an already-registered experiment (new poisoned training run):
    python run_backdoor_experiment.py \\
        --name         bd_rimone_square_v1 \\
        --dataset_name rimone \\
        --data_dir     /path/to/rimone_x

Flow
----
  1. Look up the Experiment row by name (or register it if --register is given).
  2. Read BackdoorExperimentParams for attack hyperparameters.
  3. Run poisoned federated training via backdoor_syft.run_backdoor_syft_pipeline().
  4. Register the new Split + TrainingResult in the DB and link them to the
     experiment via ExperimentSplit (result_id filled in immediately).
  5. Run Activation Clustering defense via backdoor_defense.run_defense().
  6. Persist one PoisoningRun row combining attack + defense metrics,
     linked to the poisoned model's result_id.

Prerequisites
-------------
  - migration_experiment_type.sql has been applied to the DB.
  - The Dataset row already exists (run create_dataset.py first).
  - MySQL running with credentials in database_access.DB_CONFIG.
  - PySyft, ART and PyTorch installed.
"""

import argparse
import sys

from backdoor_syft    import run_backdoor_syft_pipeline
from backdoor_defense import run_defense
from database_access import (
    register_attack_experiment,
    register_backdoor_experiment_params,
    get_experiment_by_name,
    get_backdoor_experiment_params,
    register_server,
    register_split_server,
    register_experiment_split,
    register_training_result,
    link_result_to_es,
    register_poisoning_run,
)

# Generic local server used for non-federated (single-node) pipelines.
LOCAL_SERVER_NAME  = "local"
LOCAL_SERVER_EMAIL = "local@codigla.org"
LOCAL_SERVER_PASS  = "changethis"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_new_experiment(args) -> int:
    """
    Create Experiment + BackdoorExperimentParams rows and return experiment_id.

    Training hyperparameters (lr, batch_size, epochs, patience) are stored
    in the Experiment row so they are available on re-runs without needing
    to pass them on the CLI again.
    """
    exp_id = register_attack_experiment(
        name            = args.name,
        experiment_type = "backdoor",
        description     = (
            args.description or
            f"ResNet50 backdoor | dataset={args.dataset_name} | "
            f"trigger={args.trigger_type} | poison={args.percent_poison}"
        ),
        lr          = args.lr,
        batch_size  = args.batch_size,
        epochs_max  = args.epochs,
        patience    = args.patience,
        eve_model_path = args.model_out,
    )
    print(f"[DB] Experiment registered: experiment_id={exp_id}, name='{args.name}'")

    register_backdoor_experiment_params(
        experiment_id      = exp_id,
        trigger_type       = args.trigger_type,
        percent_poison     = args.percent_poison,
        source_class       = args.source_class,
        target_class       = args.target_class,
        trigger_size       = args.trigger_size,
        trigger_position   = args.trigger_position,
        check_all_classes  = args.check_all_classes,
    )
    print("[DB] BackdoorExperimentParams saved.")

    return exp_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backdoor pipeline driven by the DB experiment registry."
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

    # Data (required every run)
    parser.add_argument("--data_dir",     required=True,
                        help="Path to the dataset directory.")
    parser.add_argument("--dataset_name", required=True,
                        help="Name of the dataset already registered in the DB.")
    parser.add_argument("--model_out",    default="backdoor_syft.pth",
                        help="Output .pth file for the poisoned model.")

    # Registration args (only needed with --register; ignored otherwise)
    parser.add_argument("--description",      default=None)
    parser.add_argument("--trigger_type",     default="square",
                        choices=["square", "cross", "checkerboard",
                                 "gaussian", "sinusoidal", "border"])
    parser.add_argument("--percent_poison",   type=float, default=0.2)
    parser.add_argument("--trigger_size",     type=int,   default=8)
    parser.add_argument("--trigger_position", default="top_left",
                        choices=["top_left", "top_right",
                                 "bottom_left", "bottom_right"])
    parser.add_argument("--source_class",     type=int,   default=1)
    parser.add_argument("--target_class",     type=int,   default=0)
    parser.add_argument("--check_all_classes", action="store_true")

    # Training hyperparameters (only needed with --register)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=500)
    parser.add_argument("--patience",    type=int,   default=50)
    parser.add_argument("--num_classes", type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.10)

    args = parser.parse_args()

    # -- Step 1: resolve experiment ------------------------------------------
    if args.register:
        exp_id = register_new_experiment(args)
    else:
        exp    = get_experiment_by_name(args.name)
        exp_id = exp["experiment_id"]
        print(f"[DB] Loaded experiment '{args.name}' -> experiment_id={exp_id}")

    # -- Step 2: load attack hyperparameters from DB -------------------------
    params = get_backdoor_experiment_params(exp_id)
    print(f"[DB] trigger_type={params['trigger_type']}  "
          f"percent_poison={params['percent_poison']}")
    print(f"     source_class={params['source_class']}  "
          f"target_class={params['target_class']}")

    # Training hyperparameters: read from Experiment row (stored at registration)
    from database_access import get_experiment
    exp_row = get_experiment(exp_id)
    lr          = exp_row["lr"]
    batch_size  = exp_row["batch_size"]
    epochs      = exp_row["epochs_max"]
    patience    = exp_row["patience"]

    # Runtime-only args not stored in DB
    seed        = args.seed
    train_ratio = args.train_ratio
    val_ratio   = args.val_ratio
    num_classes = args.num_classes

    classes_to_check = (
        None if params["check_all_classes"] else [params["target_class"]]
    )

    # -- Step 3: poisoned federated training ---------------------------------
    print("\n======================================================")
    print(" PHASE 1 -- Poisoned federated training")
    print("======================================================")
    print(f"  Trigger     : {params['trigger_type']} "
          f"(size={params['trigger_size']}, pos={params['trigger_position']})")
    print(f"  Poison rate : {params['percent_poison']}")
    print(f"  Source->tgt : {params['source_class']} -> {params['target_class']}")

    training_result = run_backdoor_syft_pipeline(
        data_dir         = args.data_dir,
        model_out        = args.model_out,
        cfg = {
            "batch_size":  batch_size,
            "epochs":      epochs,
            "lr":          lr,
            "patience":    patience,
            "num_classes": num_classes,
            "seed":        seed,
            "train_ratio": train_ratio,
            "val_ratio":   val_ratio,
        },
        trigger_type     = params["trigger_type"],
        percent_poison   = params["percent_poison"],
        trigger_size     = params["trigger_size"],
        trigger_position = params["trigger_position"],
        source_class     = params["source_class"],
        target_class     = params["target_class"],
    )

    if training_result.get("status") != "Exito":
        print("\n[ERROR] Poisoned training failed:")
        print(training_result.get("error"))
        sys.exit(1)

    # -- Step 4: register poisoned model in DB -------------------------------
    print("\n[DB] Registering poisoned training results...")

    # Ensure local server row exists (idempotent).
    register_server(
        name           = LOCAL_SERVER_NAME,
        owner_email    = LOCAL_SERVER_EMAIL,
        owner_password = LOCAL_SERVER_PASS,
    )

    split_id = register_split_server(
        dataset_name = args.dataset_name,
        server_name  = LOCAL_SERVER_NAME,
        model_path   = args.model_out,
        n_train      = training_result["train_samples"],
        n_val        = training_result["val_samples"],
        n_test       = training_result["test_samples"],
        seed         = seed,
        train_ratio  = train_ratio,
        val_ratio    = val_ratio,
    )

    es_id = register_experiment_split(exp_id, split_id)

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
    print(f"     es_id         = {es_id}")
    print(f"     result_id     = {result_id}")
    print(f"     ASR           = {training_result['attack_success_rate']:.4f}")
    print(f"     Clean bal-acc = {training_result['clean_bal_acc']:.4f}")

    # -- Step 5: Activation Clustering defense -------------------------------
    print("\n======================================================")
    print(" PHASE 2 -- Activation Clustering defense")
    print("======================================================")

    defense_result = run_defense(
        data_dir         = args.data_dir,
        model_path       = training_result["model_path"],
        trigger_type     = params["trigger_type"],
        poison_rate      = params["percent_poison"],
        trigger_size     = params["trigger_size"],
        trigger_pos      = params["trigger_position"],
        source_class     = params["source_class"],
        target_class     = params["target_class"],
        batch_size       = batch_size,
        classes_to_check = classes_to_check,
    )

    print(f"     AC precision  = {defense_result['precision']:.4f}")
    print(f"     AC recall     = {defense_result['recall']:.4f}")
    print(f"     AC F1         = {defense_result['f1']:.4f}")

    # -- Step 6: persist PoisoningRun ----------------------------------------
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
    print(" Backdoor experiment completed successfully")
    print("======================================================")


if __name__ == "__main__":
    main()
