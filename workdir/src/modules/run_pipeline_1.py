"""
run_pipeline.py
===============
Orquestador del pipeline completo de investigación de glaucoma.

  1. Recupera el dataset ya registrado en la BD (Dataset debe existir).
  2. Ejecuta el entrenamiento federado vía PySyft → guarda Split,
     Experiment, TrainingResult y ExperimentSplit en la BD.
  3. Lanza los tres ataques adversarios (FGSM, PGD, BIM) sobre el
     modelo entrenado → guarda AdversarialRun por cada ataque.

Uso:
    python run_pipeline.py \
        --data_dir     /ruta/al/dataset \
        --dataset_name RIMONE-A \
        --model_out    best_rimone_syft.pth

Requisitos previos:
    - La tabla Dataset ya tiene una fila con el nombre indicado
      en --dataset_name (se recupera su dataset_id automáticamente).
    - MySQL corriendo con las credenciales de DB_CONFIG en db_1_1.py.
    - PySyft, ART y PyTorch instalados.
"""

import argparse
import sys

from model_training_syft import run_syft_pipeline
from adversarial_attacks import run_attack
from db_1_1 import (
    get_db,
    register_split,
    register_experiment,
    register_experiment_split,
    register_training_result,
    link_result_to_es,
    register_adversarial_run,
)


# ─────────────────────────────────────────────────────────────
# Helper BD
# ─────────────────────────────────────────────────────────────

def get_dataset_id(dataset_name: str) -> int:
    """Recupera el dataset_id a partir del nombre. Lanza ValueError si no existe."""
    sql = "SELECT dataset_id FROM Dataset WHERE name = %s"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (dataset_name,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"No existe ningún dataset con nombre '{dataset_name}' en la BD. "
            "Regístralo primero con register_dataset()."
        )
    return row[0]


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo: entrenamiento federado + ataques adversarios + BD"
    )
    parser.add_argument("--data_dir",     required=True,
                        help="Ruta al directorio del dataset")
    parser.add_argument("--dataset_name", required=True,
                        help="Nombre del dataset ya registrado en la tabla Dataset")
    parser.add_argument("--model_out",    default="best_rimone_syft.pth",
                        help="Fichero .pth donde se guardará el modelo")
    # Hiperparámetros de entrenamiento
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--patience",     type=int,   default=50)
    parser.add_argument("--num_classes",  type=int,   default=2)
    # Ataques adversarios
    parser.add_argument("--epsilon",      type=float, default=0.1)
    parser.add_argument("--max_iter",     type=int,   default=10)
    parser.add_argument("--n_samples",    type=int,   default=None,
                        help="Limitar muestras por ataque (None = todas)")
    parser.add_argument("--skip_attacks", action="store_true",
                        help="Omitir la fase de ataques adversarios")
    args = parser.parse_args()

    # ── 0. Recuperar dataset_id ──────────────────────────────
    print(f"\n[BD] Buscando dataset '{args.dataset_name}'...")
    dataset_id = get_dataset_id(args.dataset_name)
    print(f"     dataset_id = {dataset_id}")

    # ────────────────────────────────────────────────────────
    # FASE 1 — Entrenamiento federado (PySyft)
    # ────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print(" FASE 1 — Entrenamiento federado (PySyft)")
    print("══════════════════════════════════════════")

    training_result = run_syft_pipeline(
        data_dir  = args.data_dir,
        model_out = args.model_out,
        cfg = {
            "batch_size":  args.batch_size,
            "epochs":      args.epochs,
            "lr":          args.lr,
            "patience":    args.patience,
            "num_classes": args.num_classes,
        },
    )

    if training_result.get("status") != "Exito":
        print("\n[ERROR] El entrenamiento falló:")
        print(training_result.get("error"))
        sys.exit(1)

    # ── Guardar en BD ────────────────────────────────────────
    print("\n[BD] Registrando resultados del entrenamiento...")

    split_id = register_split(
        dataset_id  = dataset_id,
        n_train     = training_result["train_samples"],
        n_val       = training_result["val_samples"],
        n_test      = training_result["test_samples"],
        seed        = 42,
        train_ratio = 0.70,
        val_ratio   = 0.10,
    )

    experiment_id = register_experiment(
        lr          = args.lr,
        batch_size  = args.batch_size,
        epochs_max  = args.epochs,
        patience    = args.patience,
        description = f"ResNet50 federado | dataset={args.dataset_name}",
    )

    es_id = register_experiment_split(experiment_id, split_id)

    result_id = register_training_result(
        model_path       = training_result["model_path"],
        best_epoch       = training_result["best_epoch"],
        best_val_bal_acc = training_result["best_val_bal_acc"],
        test_bal_acc     = training_result["test_bal_acc"],
        test_precision   = training_result["test_precision"],
        test_recall      = training_result["test_recall"],
    )

    link_result_to_es(es_id, result_id)

    print(f"     split_id      = {split_id}")
    print(f"     experiment_id = {experiment_id}")
    print(f"     es_id         = {es_id}")
    print(f"     result_id     = {result_id}  ✓")

    if args.skip_attacks:
        print("\n[INFO] Fase de ataques omitida (--skip_attacks).")
        print("\nPipeline completado.")
        return

    # ────────────────────────────────────────────────────────
    # FASE 2 — Ataques adversarios (ART)
    # ────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print(" FASE 2 — Ataques adversarios (ART)")
    print("══════════════════════════════════════════")

    attacks = [
        {
            "attack_type": "fgsm",
            "epsilon":     args.epsilon,
            "eps_step":    None,
            "max_iter":    args.max_iter,
            "num_random_init": 1,
        },
        {
            "attack_type": "pgd",
            "epsilon":     args.epsilon,
            "eps_step":    args.epsilon / 4,
            "max_iter":    args.max_iter,
            "num_random_init": 1,
        },
        {
            "attack_type": "bim",
            "epsilon":     args.epsilon,
            "eps_step":    args.epsilon / 4,
            "max_iter":    args.max_iter,
            "num_random_init": 1,
        },
    ]

    for atk in attacks:
        atype = atk["attack_type"].upper()
        print(f"\n── Ataque {atype} ──────────────────────────")
        try:
            metrics = run_attack(
                data_dir        = args.data_dir,
                model_path      = training_result["model_path"],
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

            print(f"  [BD] AdversarialRun guardado → adv_id={adv_id}")
            print(f"       Bal-Acc limpio  : {metrics['clean_bal_acc']:.4f}")
            print(f"       Bal-Acc adverso : {metrics['adv_bal_acc']:.4f}  "
                  f"(caída={metrics['clean_bal_acc'] - metrics['adv_bal_acc']:.4f})")

        except Exception as exc:
            print(f"  [WARN] Ataque {atype} falló y se omite: {exc}")

    print("\n══════════════════════════════════════════")
    print(" Pipeline completado con éxito ✓")
    print("══════════════════════════════════════════")


if __name__ == "__main__":
    main()
