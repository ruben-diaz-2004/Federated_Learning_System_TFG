"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================
run_backdoor_pipeline.py
========================
Orquestador del pipeline de entrenamiento con BACKDOOR sobre PySyft +
defensa por Activation Clustering + persistencia en BD.

Flujo:
  1. Recupera el dataset ya registrado en la BD (Dataset debe existir).
  2. Ejecuta el entrenamiento federado envenenado vía PySyft
     (backdoor_syft.run_backdoor_syft_pipeline) → guarda Split, Experiment,
     TrainingResult y ExperimentSplit en la BD.
  3. Ejecuta la defensa Activation Clustering sobre el modelo envenenado
     (backdoor_defense.run_defense) → obtiene ac_precision/ac_recall/ac_f1.
  4. Inserta una única fila en PoisoningRun con todo: ataque + defensa.

Uso:
    python run_backdoor_pipeline.py \
        --data_dir       /ruta/al/dataset \
        --dataset_name   RIMONE \
        --model_out      backdoor_syft.pth \
        --trigger_type   square \
        --percent_poison 0.2
"""

import argparse
import sys

from backdoor_syft    import run_backdoor_syft_pipeline
from backdoor_defense import run_defense
from database_access import (
    get_db,
    register_split,
    register_experiment,
    register_experiment_split,
    register_training_result,
    link_result_to_es,
    register_poisoning_run,
)


def get_dataset_id(dataset_name: str) -> int:
    """Recupera el dataset_id a partir del nombre. ValueError si no existe."""
    sql = "SELECT dataset_id FROM Dataset WHERE name = %s"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (dataset_name,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"No existe ningún dataset con nombre '{dataset_name}' en la BD. "
            "Regístralo primero con register_dataset() / create_dataset.py."
        )
    return row[0]


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline backdoor: entrenamiento envenenado + defensa AC + BD",
    )
    # Dataset
    parser.add_argument("--data_dir",     required=True,
                        help="Ruta al directorio del dataset")
    parser.add_argument("--dataset_name", required=True,
                        help="Nombre del dataset ya registrado en Dataset")
    parser.add_argument("--model_out",    default="backdoor_syft.pth",
                        help="Fichero .pth del modelo envenenado")

    # Hiperparámetros de entrenamiento
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--patience",     type=int,   default=50)
    parser.add_argument("--num_classes",  type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--train_ratio",  type=float, default=0.70)
    parser.add_argument("--val_ratio",    type=float, default=0.10)

    # Backdoor
    parser.add_argument("--trigger_type",     type=str, default="square",
                        choices=["square", "cross", "checkerboard",
                                 "gaussian", "sinusoidal", "border"])
    parser.add_argument("--percent_poison",   type=float, default=0.2)
    parser.add_argument("--trigger_size",     type=int,   default=8)
    parser.add_argument("--trigger_position", type=str,   default="top_left",
                        choices=["top_left", "top_right",
                                 "bottom_left", "bottom_right"])
    parser.add_argument("--source_class",     type=int,   default=1)
    parser.add_argument("--target_class",     type=int,   default=0)

    # Defensa
    parser.add_argument("--check_all_classes", action="store_true",
                        help="AC sobre todas las clases en vez de solo target")

    args = parser.parse_args()

    # ── 0. Recuperar dataset_id ──────────────────────────────
    print(f"\n[BD] Buscando dataset '{args.dataset_name}'...")
    dataset_id = get_dataset_id(args.dataset_name)
    print(f"     dataset_id = {dataset_id}")

    # ══════════════════════════════════════════════════════════
    # FASE 1 — Entrenamiento federado con backdoor
    # ══════════════════════════════════════════════════════════
    print(" FASE 1 — Entrenamiento federado con backdoor")
    print(f"  Trigger     : {args.trigger_type} "
          f"(size={args.trigger_size}, pos={args.trigger_position})")
    print(f"  % poison    : {args.percent_poison}")
    print(f"  Source → tgt: {args.source_class} → {args.target_class}")

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
        print("\n[ERROR] El entrenamiento envenenado falló:")
        print(training_result.get("error"))
        sys.exit(1)

    split_id = register_split(
        dataset_id  = dataset_id,
        n_train     = training_result["train_samples"],
        n_val       = training_result["val_samples"],
        n_test      = training_result["test_samples"],
        seed        = args.seed,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
    )

    experiment_id = register_experiment(
        lr          = args.lr,
        batch_size  = args.batch_size,
        epochs_max  = args.epochs,
        patience    = args.patience,
        description = (
            f"ResNet50 federado con backdoor | dataset={args.dataset_name} | "
            f"trigger={args.trigger_type} | percent_poison={args.percent_poison}"
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

    # ══════════════════════════════════════════════════════════
    # FASE 2 — Defensa Activation Clustering
    # ══════════════════════════════════════════════════════════
    print(" FASE 2 — Defensa Activation Clustering")
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

    print(" FASE 3 — Insert en PoisoningRun")

    poison_id = register_poisoning_run(
        result_id           = result_id,
        trigger_type        = training_result["trigger_type"],
        trigger_size        = training_result["trigger_size"],       # None en sinusoidal
        trigger_position    = training_result["trigger_position"],   # None en sinusoidal
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
    print(f"     ASR           = {training_result['attack_success_rate']:.4f}  "
          f"(efectividad del ataque)")
    print(f"     AC F1         = {defense_result['f1']:.4f}  "
          f"(efectividad del defensor)")
    print(f"     Test bal-acc  = {training_result['clean_bal_acc']:.4f}  "
          f"(sobre test limpio)")

    print(" Pipeline completado con éxito")

if __name__ == "__main__":
    main()
