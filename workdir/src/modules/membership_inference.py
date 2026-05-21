"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================
membership_inference.py

Ataques de inferencia de pertenencia (Membership Inference Attack, MIA)
sobre ResNet50 con ART (Adversarial Robustness Toolbox).

Variantes soportadas:
  - rule_based : umbral sobre la confianza máxima del modelo (sin entrenamiento
                 del atacante). Sirve como baseline.
  - rf         : Random Forest entrenado sobre los vectores de probabilidad.
  - nn         : Red neuronal (MLP) entrenada como clasificador de membresía.
  - gb         : Gradient Boosting (más potente, más lento).

Métrica principal:
  MIA accuracy — fracción de muestras correctamente clasificadas como
  miembro/no-miembro. Valor esperado bajo privacidad: ~0.50 (azar).
  Valores > 0.60 indican fuga de privacidad significativa.

Uso:
    python membership_inference.py \
        --model_path best_resnet50.pth \
        --data_dir /ruta/rimone_A \
        --variant rf

    # Todas las variantes de una vez:
    python membership_inference.py \
        --model_path best_resnet50.pth \
        --data_dir /ruta/rimone_A \
        --variant all
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)

# ART — clasificador PyTorch (reutilizamos el mismo wrapper que en adversarial_attacks.py)
from art.estimators.classification import PyTorchClassifier

# ART — ataques de inferencia de pertenencia
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    MembershipInferenceBlackBoxRuleBased,
)

from data_preprocessing import Data_Preprocessing
from train_resnet import (
    build_resnet50, split_dataset, collate_fn,
    ValProcessor, HFTransform, SEED, set_seed,
)
from adversarial_attacks import (
    build_art_classifier,   # reutilizamos el wrapper ART ya definido
    dataset_to_numpy,       # reutilizamos la extracción numpy
)

# Variantes disponibles (excluye 'all', que es un alias especial)
SUPPORTED_VARIANTS = ("rule_based", "rf", "nn", "gb")


# ----------------------------------------------
# Creación del ataque MIA
# ----------------------------------------------

def create_mia(variant: str, classifier: PyTorchClassifier):
    """
    Instancia el objeto de ataque MIA según la variante indicada.

    variant:
        'rule_based' — MembershipInferenceBlackBoxRuleBased: clasifica como
                       miembro toda muestra cuya confianza máxima supere un
                       umbral (por defecto 0.5). No requiere entrenamiento del
                       atacante; es el baseline más simple.
        'rf'         — Random Forest sobre el vector de softmax. Equilibrio
                       entre coste y potencia.
        'nn'         — MLP (red neuronal) como clasificador de membresía.
                       Más expresivo que RF pero más costoso.
        'gb'         — Gradient Boosting. Generalmente el más potente de los
                       tres modelos de ataque.

    Devuelve (attack_object, needs_fit: bool).
    rule_based no necesita fit(); los demás sí.
    """
    if variant == "rule_based":
        return MembershipInferenceBlackBoxRuleBased(classifier), False

    if variant in ("rf", "nn", "gb"):
        return MembershipInferenceBlackBox(
            classifier, attack_model_type=variant
        ), True

    raise ValueError(
        f"Variante '{variant}' no soportada. Elige entre: "
        + ", ".join(SUPPORTED_VARIANTS)
    )


# ----------------------------------------------
# Evaluación de un ataque MIA ya inferido
# ----------------------------------------------

def evaluate_mia(inferred_train: np.ndarray,
                 inferred_test: np.ndarray,
                 variant: str) -> dict:
    """
    Calcula las métricas del ataque a partir de los arrays binarios
    devueltos por attack.infer().

    Convención:
        1 → el atacante predice "miembro del training set"
        0 → el atacante predice "no miembro"

    Devuelve un dict con:
        mia_accuracy, mia_precision, mia_recall,
        mia_auc, member_rate, non_member_rate,
        confusion_matrix (lista 2×2 serializable).
    """
    n_train = len(inferred_train)
    n_test  = len(inferred_test)

    # Ground truth: train → 1, test → 0
    y_true = np.concatenate([
        np.ones(n_train,  dtype=np.int32),
        np.zeros(n_test,  dtype=np.int32),
    ])
    y_pred = np.concatenate([inferred_train, inferred_test]).astype(np.int32)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred).tolist()

    # AUC solo si hay variación en las predicciones (evita error con ROC plano)
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        auc = float("nan")

    # Tasas de acierto por grupo
    member_rate     = float(np.mean(inferred_train == 1))
    non_member_rate = float(np.mean(inferred_test  == 0))

    print(f"\n  -- Resultados MIA [{variant}] ")
    print(f"  MIA Accuracy   : {accuracy:.4f}  "
          f"(baseline aleatorio = 0.5000)")
    print(f"  MIA Precision  : {precision:.4f}")
    print(f"  MIA Recall     : {recall:.4f}")
    print(f"  MIA AUC        : {auc:.4f}")
    print(f"  Tasa miembro identificado    : {member_rate:.4f}  "
          f"({int(member_rate * n_train)}/{n_train})")
    print(f"  Tasa no-miembro identificado : {non_member_rate:.4f}  "
          f"({int(non_member_rate * n_test)}/{n_test})")
    print(f"  Confusion matrix:\n  {cm}")

    privacy_risk = (
        "ALTO"   if accuracy > 0.70 else
        "MEDIO"  if accuracy > 0.60 else
        "BAJO"
    )
    print(f"  → Riesgo de privacidad: {privacy_risk}")

    return {
        "variant":          variant,
        "mia_accuracy":     round(accuracy,  4),
        "mia_precision":    round(precision, 4),
        "mia_recall":       round(recall,    4),
        "mia_auc":          round(auc,       4) if not np.isnan(auc) else None,
        "member_rate":      round(member_rate,     4),
        "non_member_rate":  round(non_member_rate, 4),
        "n_train_samples":  n_train,
        "n_test_samples":   n_test,
        "confusion_matrix": cm,
    }


# ----------------------------------------------
# Función principal del ataque MIA
# ----------------------------------------------

def run_mia(data_dir:    str,
            model_path:  str,
            variant:     str  = "rf",
            batch_size:  int  = 32,
            n_train_max: int  = None,
            n_test_max:  int  = None) -> dict:
    """
    Ejecuta un Membership Inference Attack sobre el modelo entrenado.

    Parámetros
    ----------
    data_dir    : ruta al directorio del dataset (mismo que en train_resnet.py).
    model_path  : ruta al .pth del modelo entrenado.
    variant     : variante del ataque ('rule_based', 'rf', 'nn', 'gb').
    batch_size  : tamaño de batch para la extracción numpy.
    n_train_max : limitar el nº de muestras de train (None = todas).
                  Útil para acelerar el entrenamiento del clasificador atacante.
    n_test_max  : ídem para test.

    Retorna
    -------
    dict con las métricas del ataque (ver evaluate_mia).
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # -- 1. Cargar y dividir dataset ------------------------------------------
    print("\n-- Cargando dataset --")
    data_prep    = Data_Preprocessing(
        data_path=Path(data_dir), prep_batch_size=batch_size
    )
    base_dataset = data_prep.dataset
    train_idx, val_idx, test_idx = split_dataset(base_dataset)

    # Usamos ValProcessor en ambos splits para evitar augmentación
    train_split = base_dataset.select(train_idx)
    test_split  = base_dataset.select(test_idx)
    train_split.set_transform(HFTransform(ValProcessor()))
    test_split.set_transform(HFTransform(ValProcessor()))

    # -- 2. Convertir a numpy -------------------------------------------------
    print("\n-- Extrayendo arrays numpy --")
    x_train, y_train = dataset_to_numpy(train_split, batch_size)
    x_test,  y_test  = dataset_to_numpy(test_split,  batch_size)

    # Subconjunto opcional para acelerar ataques costosos (rf, nn, gb)
    if n_train_max and n_train_max < len(x_train):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(x_train), size=n_train_max, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
        print(f"  Usando {n_train_max} muestras de train (de {len(train_idx)})")

    if n_test_max and n_test_max < len(x_test):
        rng = np.random.default_rng(SEED + 1)
        idx = rng.choice(len(x_test), size=n_test_max, replace=False)
        x_test, y_test = x_test[idx], y_test[idx]
        print(f"  Usando {n_test_max} muestras de test  (de {len(test_idx)})")

    print(f"  x_train: {x_train.shape}  x_test: {x_test.shape}")

    # -- 3. Cargar modelo y construir clasificador ART ------------------------
    print("\n-- Cargando modelo --")
    model = build_resnet50(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    classifier = build_art_classifier(model, device)

    # -- 4. Crear y (opcionalmente) entrenar el atacante ----------------------
    print(f"\n-- Configurando ataque MIA [{variant}] --")
    attack, needs_fit = create_mia(variant, classifier)

    if needs_fit:
        # El clasificador de ataque se entrena con la mitad de cada split
        # para que la evaluación se haga sobre muestras no vistas por el atacante.
        n_fit_train = max(1, len(x_train) // 2)
        n_fit_test  = max(1, len(x_test)  // 2)

        print(f"  Entrenando clasificador atacante con "
              f"{n_fit_train} muestras train + {n_fit_test} muestras test...")
        attack.fit(
            x_train[:n_fit_train], y_train[:n_fit_train],
            x_test[:n_fit_test],   y_test[:n_fit_test],
        )
        print("  Clasificador atacante entrenado.")

        # Evaluamos sobre la otra mitad (muestras no vistas en el fit)
        x_train_eval = x_train[n_fit_train:]
        y_train_eval = y_train[n_fit_train:]
        x_test_eval  = x_test[n_fit_test:]
        y_test_eval  = y_test[n_fit_test:]
    else:
        # rule_based no necesita fit; evaluamos sobre todo el conjunto
        x_train_eval, y_train_eval = x_train, y_train
        x_test_eval,  y_test_eval  = x_test,  y_test

    # -- 5. Inferencia de membresía -------------------------------------------
    print("\n-- Ejecutando inferencia de membresía --")
    inferred_train = attack.infer(x_train_eval, y_train_eval)
    inferred_test  = attack.infer(x_test_eval,  y_test_eval)

    # -- 6. Métricas ----------------------------------------------------------
    metrics = evaluate_mia(inferred_train, inferred_test, variant)

    return metrics


# ----------------------------------------------
# Ejecución de todas las variantes
# ----------------------------------------------

def run_all_variants(data_dir:   str,
                     model_path: str,
                     batch_size: int = 32,
                     n_train_max: int = None,
                     n_test_max:  int = None) -> dict:
    """
    Ejecuta las cuatro variantes MIA y devuelve un dict
    {variant: metrics_dict} para facilitar la comparación.
    """
    results = {}
    for variant in SUPPORTED_VARIANTS:
        print(f"\n{'═' * 50}")
        print(f"  Variante: {variant.upper()}")
        print(f"{'═' * 50}")
        try:
            results[variant] = run_mia(
                data_dir    = data_dir,
                model_path  = model_path,
                variant     = variant,
                batch_size  = batch_size,
                n_train_max = n_train_max,
                n_test_max  = n_test_max,
            )
        except Exception as exc:
            print(f"  [WARN] Variante '{variant}' falló y se omite: {exc}")
            results[variant] = {"variant": variant, "error": str(exc)}

    # Resumen comparativo
    print(f"\n{'═' * 50}")
    print("  RESUMEN COMPARATIVO")
    print(f"{'═' * 50}")
    print(f"  {'Variante':<12} {'Accuracy':>10} {'AUC':>8} {'Riesgo'}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
    for v, m in results.items():
        if "error" in m:
            print(f"  {v:<12}  ERROR: {m['error'][:40]}")
            continue
        acc  = m.get("mia_accuracy", float("nan"))
        auc  = m.get("mia_auc") or float("nan")
        risk = "ALTO" if acc > 0.70 else "MEDIO" if acc > 0.60 else "BAJO"
        print(f"  {v:<12} {acc:>10.4f} {auc:>8.4f} {risk}")

    return results


# ----------------------------------------------
# Main
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Membership Inference Attack sobre ResNet50 - Glaucoma"
    )
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="Ruta al directorio del dataset")
    parser.add_argument("--model_path",  type=str, required=True,
                        help="Ruta al .pth del modelo entrenado")
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

    if args.variant == "all":
        run_all_variants(
            data_dir    = args.data_dir,
            model_path  = args.model_path,
            batch_size  = args.batch_size,
            n_train_max = args.n_train_max,
            n_test_max  = args.n_test_max,
        )
    else:
        run_mia(
            data_dir    = args.data_dir,
            model_path  = args.model_path,
            variant     = args.variant,
            batch_size  = args.batch_size,
            n_train_max = args.n_train_max,
            n_test_max  = args.n_test_max,
        )


if __name__ == "__main__":
    main()
