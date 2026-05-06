"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================
backdoor_defense.py
===================
Defensa contra backdoors mediante Activation Clustering (Chen et al. 2018).

Idea: un modelo backdoored desarrolla "neuronas trigger" que se activan
de forma anómala con las muestras envenenadas. Si extraemos las activaciones
de la última capa oculta y aplicamos clustering por clase, las muestras
envenenadas tienden a caer en un cluster minoritario separado.

Flujo:
  1. Reconstruir el training set envenenado (mismo SEED → mismo chosen_idx).
  2. Extraer activaciones de la penúltima capa (vector de 128) con forward hook.
  3. Por cada clase: PCA(10) → KMeans(k=2).
  4. Marcar como sospechoso el cluster minoritario.
  5. Comparar contra chosen_idx (ground truth) → precision/recall/F1.

Uso:
    python backdoor_defense.py \
        --data_dir ../../rimone_A \
        --model_path backdoor_resnet50.pth \
        --trigger_type square \
        --poison_rate 0.2 \
        --source_class 1 \
        --target_class 0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix)

from data_preprocessing import Data_Preprocessing
from train_resnet import (set_seed, build_resnet50, split_dataset, HFTransform, TrainProcessor, SEED)
from backdoor_attack import (get_trigger, dataset_to_numpy, poison_dataset, make_numpy_loader)


def extract_activations(model, x, batch_size, device, layer_idx=4):
    """
    Pasa todas las muestras x por el modelo y devuelve las activaciones
    de la capa indicada (por defecto layer 4 = ReLU tras Linear(2048, 128)).

    Devuelve array (N, D) donde D es la dimensionalidad de la capa (128).
    """
    target_layer = model[layer_idx]
    activations = []

    def hook(module, inp, out):
        # out shape: (B, D); lo movemos a CPU y numpy directamente
        activations.append(out.detach().cpu().numpy())

    handle = target_layer.register_forward_hook(hook)

    model.eval()
    loader = make_numpy_loader(x, np.zeros(len(x), dtype=np.int64), batch_size)
    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))

    handle.remove()
    return np.concatenate(activations, axis=0)



def cluster_class_activations(activations, n_components=10, random_state=SEED):
    """
    Aplica PCA(n_components) + KMeans(k=2) sobre las activaciones de UNA clase.
    Devuelve:
        labels        : (N,) cluster asignado a cada muestra (0 ó 1)
        suspect_label : int, etiqueta del cluster sospechoso (el más pequeño)
        sizes         : (size_cluster_0, size_cluster_1)
    """
    n_samples = len(activations)
    # PCA necesita n_components <= min(n_samples, n_features)
    n_comp = min(n_components, n_samples - 1, activations.shape[1])

    pca = PCA(n_components=n_comp, random_state=random_state)
    reduced = pca.fit_transform(activations)

    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = km.fit_predict(reduced)

    sizes = (int(np.sum(labels == 0)), int(np.sum(labels == 1)))
    # Criterio "size": el cluster minoritario es el sospechoso
    suspect_label = int(np.argmin(sizes))
    return labels, suspect_label, sizes


def detect_poisoned_samples(activations, y, classes_to_check=None):
    """
    Aplica clustering por clase y devuelve un array booleano (N,)
    donde True = muestra marcada como sospechosa por el defensor.

    classes_to_check: si se da, solo se analizan esas clases.
                      Por defecto, todas las clases presentes en y.
    """
    n = len(activations)
    is_suspicious = np.zeros(n, dtype=bool)

    if classes_to_check is None:
        classes_to_check = np.unique(y)

    cluster_info = {}  # diagnóstico por clase

    for cls in classes_to_check:
        mask = (y == cls)
        idx_cls = np.where(mask)[0]
        if len(idx_cls) < 4:
            # pocas muestras para clusterizar de forma estable
            cluster_info[int(cls)] = {"skipped": True, "n": int(len(idx_cls))}
            continue

        labels_cls, suspect, sizes = cluster_class_activations(activations[idx_cls])
        suspect_idx = idx_cls[labels_cls == suspect]
        is_suspicious[suspect_idx] = True

        cluster_info[int(cls)] = {
            "skipped":      False,
            "n":            int(len(idx_cls)),
            "cluster_0":    sizes[0],
            "cluster_1":    sizes[1],
            "suspect":      suspect,
            "n_flagged":    int(len(suspect_idx)),
        }

    return is_suspicious, cluster_info


# ──────────────────────────────────────────────
# Métricas del detector contra ground truth
# ──────────────────────────────────────────────
def evaluate_detector(is_suspicious, chosen_idx, n_total):
    """
    Compara las predicciones del defensor con la ground truth (chosen_idx).

    Devuelve dict con tp, fp, tn, fn, precision, recall, f1.
    """
    y_true = np.zeros(n_total, dtype=bool)
    y_true[chosen_idx] = True
    y_pred = is_suspicious

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


# ──────────────────────────────────────────────
# Pipeline principal de defensa
# ──────────────────────────────────────────────
def run_defense(
    data_dir:     str,
    model_path:   str,
    trigger_type: str,
    poison_rate:  float = 0.2,
    trigger_size: int   = 8,
    trigger_pos:  str   = "top_left",
    source_class: int   = 1,
    target_class: int   = 0,
    batch_size:   int   = 32,
    classes_to_check: list = None,
):
    """
    Reconstruye el training set envenenado, extrae activaciones del modelo
    backdoored y aplica Activation Clustering para detectar las muestras
    envenenadas. Compara contra la ground truth.

    classes_to_check: por defecto solo analiza target_class (donde el atacante
                      ha mezclado muestras envenenadas con muestras genuinas).
                      Pasar None implica analizar todas las clases.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── 1. Reconstruir training set envenenado (idéntico al del ataque) ──
    print("\n── Reconstruyendo training set envenenado ──")
    base_dataset = Data_Preprocessing(
        data_path=Path(data_dir), prep_batch_size=batch_size
    ).dataset
    train_idx, _, _ = split_dataset(base_dataset)
    train_split = base_dataset.select(train_idx)
    train_split.set_transform(HFTransform(TrainProcessor()))

    x_train, y_train = dataset_to_numpy(train_split, batch_size)

    trigger_fn = get_trigger(trigger_type, size=trigger_size, position=trigger_pos)
    x_train_p, y_train_p, n_poisoned, chosen_idx = poison_dataset(
        x_train, y_train,
        trigger_fn=trigger_fn,
        poison_rate=poison_rate,
        source_class=source_class,
        target_class=target_class,
    )
    print(f"  Total train: {len(x_train_p)} | Envenenadas (GT): {n_poisoned}")

    # ── 2. Cargar modelo backdoored ──
    print("\n── Cargando modelo backdoored ──")
    model = build_resnet50(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ── 3. Extraer activaciones de la penúltima capa ──
    print("\n── Extrayendo activaciones (capa 4: ReLU tras Linear(2048,128)) ──")
    activations = extract_activations(model, x_train_p, batch_size, device, layer_idx=4)
    print(f"  Activaciones: {activations.shape}")

    # ── 4. Clustering por clase ──
    print("\n── Activation Clustering por clase ──")
    if classes_to_check is None:
        # Por defecto solo target_class: ahí están mezcladas envenenadas + limpias
        classes_to_check = [target_class]
    print(f"  Clases analizadas: {classes_to_check}")

    is_suspicious, info = detect_poisoned_samples(
        activations, y_train_p, classes_to_check=classes_to_check
    )

    for cls, data in info.items():
        if data["skipped"]:
            print(f"  Clase {cls}: omitida (n={data['n']} demasiado pequeño)")
        else:
            print(f"  Clase {cls}: n={data['n']} | "
                  f"clusters=({data['cluster_0']}, {data['cluster_1']}) | "
                  f"sospechoso=cluster_{data['suspect']} ({data['n_flagged']} muestras)")

    # ── 5. Evaluación contra ground truth ──
    print("\n── Métricas del detector contra ground truth ──")
    metrics = evaluate_detector(is_suspicious, chosen_idx, len(x_train_p))

    print(f"  TP (envenenadas detectadas)  : {metrics['tp']:4d}")
    print(f"  FN (envenenadas no detectadas): {metrics['fn']:4d}")
    print(f"  FP (limpias mal marcadas)    : {metrics['fp']:4d}")
    print(f"  TN (limpias correctas)       : {metrics['tn']:4d}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")

    # ── Resumen ──
    print("\n══ Resumen de la defensa ══")
    print(f"  Trigger             : {trigger_type}")
    print(f"  Muestras envenenadas: {n_poisoned} de {len(x_train_p)}")
    print(f"  Detectadas (recall) : {metrics['recall']:.2%}")
    print(f"  Precisión           : {metrics['precision']:.2%}")
    print(f"  F1                  : {metrics['f1']:.4f}")

    return {
        "trigger_type":  trigger_type,
        "n_total":       int(len(x_train_p)),
        "n_poisoned":    int(n_poisoned),
        "n_flagged":     int(is_suspicious.sum()),
        **metrics,
        "cluster_info":  info,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Defensa Activation Clustering contra backdoors."
    )
    parser.add_argument("--data_dir",     type=str,   required=True)
    parser.add_argument("--model_path",   type=str,   required=True,
                        help="Ruta al .pth del modelo backdoored")
    parser.add_argument("--trigger_type", type=str,   required=True,
                        choices=["square", "cross", "checkerboard",
                                 "gaussian", "sinusoidal", "border"])
    parser.add_argument("--poison_rate",  type=float, default=0.2)
    parser.add_argument("--trigger_size", type=int,   default=8)
    parser.add_argument("--trigger_pos",  type=str,   default="top_left",
                        choices=["top_left", "top_right",
                                 "bottom_left", "bottom_right"])
    parser.add_argument("--source_class", type=int,   default=1)
    parser.add_argument("--target_class", type=int,   default=0)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--check_all_classes", action="store_true",
                        help="Analizar todas las clases, no solo target_class")
    args = parser.parse_args()

    classes_to_check = None if args.check_all_classes else [args.target_class]

    run_defense(
        data_dir         = args.data_dir,
        model_path       = args.model_path,
        trigger_type     = args.trigger_type,
        poison_rate      = args.poison_rate,
        trigger_size     = args.trigger_size,
        trigger_pos      = args.trigger_pos,
        source_class     = args.source_class,
        target_class     = args.target_class,
        batch_size       = args.batch_size,
        classes_to_check = classes_to_check,
    )


if __name__ == "__main__":
    main()
