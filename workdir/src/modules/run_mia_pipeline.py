"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================
run_mia_pipeline.py
===================
Orquestador del Membership Inference Attack (MIA) sobre un modelo ya entrenado.

  1. Recupera el result_id del entrenamiento (debe existir en TrainingResult).
  2. Ejecuta una o todas las variantes del ataque MIA.
  3. Registra cada ejecución en MembershipInferenceRun.

Uso típico — una sola variante:
    python run_mia_pipeline.py \
        --data_dir   /ruta/al/dataset \
        --result_id  3 \
        --variant    rf

Uso típico — todas las variantes:
    python run_mia_pipeline.py \
        --data_dir   /ruta/al/dataset \
        --result_id  3 \
        --variant    all

El model_path se recupera automáticamente del TrainingResult indicado;
Se puede sobreescribir con --model_path si necesitas atacar un .pth distinto.
"""

import argparse
import sys

from membership_inference import run_mia, run_all_variants, SUPPORTED_VARIANTS
from database_access import get_db, register_mia_run


def get_model_path_from_result(result_id: int) -> str:
    """
    Recupera el model_path asociado a un TrainingResult.
    Lanza ValueError si el result_id no existe.
    """
    sql = "SELECT model_path FROM TrainingResult WHERE result_id = %s"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (result_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"No existe ningún TrainingResult con result_id={result_id}. "
            "Ejecuta primero run_pipeline.py para entrenar el modelo."
        )
    return row[0]


def save_mia_to_db(result_id: int, metrics: dict) -> int:
    """
    Persiste un dict de métricas (devuelto por run_mia) en la tabla
    MembershipInferenceRun. Devuelve el mia_id generado.
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
    print(f"  [BD] MembershipInferenceRun guardado → mia_id={mia_id} "
          f"(variant={metrics['variant']}, accuracy={metrics['mia_accuracy']:.4f})")
    return mia_id


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline MIA: ejecución de Membership Inference Attack + persistencia en BD"
    )
    parser.add_argument("--data_dir",    required=True,
                        help="Ruta al directorio del dataset")
    parser.add_argument("--result_id",   type=int, required=True,
                        help="result_id del TrainingResult sobre el que atacar")
    parser.add_argument("--model_path",  default=None,
                        help="Ruta al .pth del modelo (opcional; por defecto se "
                             "recupera el almacenado en TrainingResult)")
    parser.add_argument("--variant",     type=str, default="rf",
                        choices=list(SUPPORTED_VARIANTS) + ["all"],
                        help="Variante del ataque (default: rf). "
                             "'all' ejecuta las cuatro variantes.")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--n_train_max", type=int, default=None,
                        help="Limitar nº muestras de train usadas (None = todas)")
    parser.add_argument("--n_test_max",  type=int, default=None,
                        help="Limitar nº muestras de test  usadas (None = todas)")
    args = parser.parse_args()

    # ── 0. Resolver model_path desde la BD ───────────────────────────────────
    print(f"\n[BD] Recuperando model_path para result_id={args.result_id}...")
    db_model_path = get_model_path_from_result(args.result_id)
    print(f"     model_path en BD : {db_model_path}")

    if args.model_path is None:
        model_path = db_model_path
    else:
        model_path = args.model_path
        if model_path != db_model_path:
            print(f"  [WARN] --model_path ({model_path}) no coincide con el de BD. "
                  f"Usando el indicado por argumento.")

    print(" Membership Inference Attack")

    # ── 1. Ejecutar ataque(s) ────────────────────────────────────────────────
    if args.variant == "all":
        results = run_all_variants(
            data_dir    = args.data_dir,
            model_path  = model_path,
            batch_size  = args.batch_size,
            n_train_max = args.n_train_max,
            n_test_max  = args.n_test_max,
        )

        # ── 2. Persistir cada variante en BD ─────────────────────────────────
        saved = 0
        for variant, metrics in results.items():
            if "error" in metrics:
                print(f"  [SKIP] {variant}: ataque falló, no se registra en BD")
                continue
            try:
                save_mia_to_db(args.result_id, metrics)
                saved += 1
            except Exception as exc:
                print(f"  [WARN] No se pudo guardar la variante '{variant}' en BD: {exc}")

        print(f"\n  Total variantes guardadas: {saved}/{len(SUPPORTED_VARIANTS)}")

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
            save_mia_to_db(args.result_id, metrics)
        except Exception as exc:
            print(f"  [ERROR] No se pudo guardar en BD: {exc}")
            sys.exit(1)

    print(" Pipeline MIA completado con éxito")

if __name__ == "__main__":
    main()
