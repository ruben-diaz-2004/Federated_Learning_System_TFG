"""
Paso 3: Ataque de envenenamiento de datos (data poisoning) mediante imágenes adversarias
=========================================================================

Escenario:
  - Institución A  →  datos RIM-ONE,  entrena modelos
  - Institución X  →  datos Refuge,   es el atacante

Flujo completo
--------------
  1. A entrena un ResNet50 con sus datos RIM-ONE  →  modelo_A.pth
     (este paso se hace con train_resnet50.py; aquí solo se carga)

  2. X usa modelo_A y sus propios datos Refuge para generar imágenes
     adversarias con FGSM / PGD / BIM.

  3. X envía esas imágenes adversarias a A.

  4. A entrena un nuevo ResNet50 mezclando sus imágenes originales
     con un porcentaje creciente de adversarias:
         ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

  5. Evaluamos el modelo resultante en el conjunto de test LIMPIO de A
     y registramos la degradación de balanced accuracy, precision y recall.

Uso:
    python poisoning_experiment.py \
        --data_dir_A  /ruta/rimone  \
        --data_dir_X  /ruta/refuge  \
        --model_A     best_resnet50.pth \
        --attack      fgsm \
        --epsilon     0.02
"""

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, ConcatDataset
from torchvision import transforms
from torchvision.transforms.v2 import Compose, RandomCrop, RandomHorizontalFlip, \
    RandomVerticalFlip, ColorJitter, Normalize, ToTensor
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from pathlib import Path
import argparse

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod

from data_preprocessing import Data_Preprocessing
from train_resnet50 import (
    build_resnet50, split_dataset, collate_fn,
    TrainProcessor, ValProcessor, HFTransform,
    SEED, set_seed, IMAGENET_MEAN, IMAGENET_STD
)

# ──────────────────────────────────────────────
# Helpers: carga de datos
# ──────────────────────────────────────────────
def load_institution_data(data_dir, batch_size):
    """
    Carga el dataset de una institución con Data_Preprocessing y devuelve
    los splits train/val/test como HF datasets con sus transforms aplicados.
    """
    data_prep = Data_Preprocessing(
        data_path=Path(data_dir),
        image_size=[256, 256],
        image_processor=None,
        num_proc=1,
        prep_batch_size=batch_size,
    )
    base = data_prep.dataset
    train_idx, val_idx, test_idx = split_dataset(base)

    train_split = base.select(train_idx)
    val_split   = base.select(val_idx)
    test_split  = base.select(test_idx)

    train_split.set_transform(HFTransform(TrainProcessor()))
    val_split.set_transform(HFTransform(ValProcessor()))
    test_split.set_transform(HFTransform(ValProcessor()))

    return train_split, val_split, test_split, train_idx, base


def hf_split_to_numpy(hf_split, batch_size=32):
    """Extrae todos los tensores del split → numpy (N, C, H, W) float32."""
    loader = DataLoader(hf_split, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    xs, ys = [], []
    for imgs, labels in loader:
        xs.append(imgs.numpy())
        ys.append(labels.numpy())
    return (np.concatenate(xs, axis=0).astype(np.float32),
            np.concatenate(ys, axis=0).astype(np.int64))


# ──────────────────────────────────────────────
# Wrapper ART
# ──────────────────────────────────────────────
def build_art_classifier(model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=2,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


# ──────────────────────────────────────────────
# Creación del ataque
# ──────────────────────────────────────────────
def create_attack(attack_type, classifier, epsilon, eps_step, max_iter, num_random_init):
    if eps_step is None:
        eps_step = epsilon / 4.0
    if attack_type == 'fgsm':
        return FastGradientMethod(estimator=classifier, eps=epsilon)
    elif attack_type == 'pgd':
        return ProjectedGradientDescent(
            estimator=classifier, eps=epsilon, eps_step=eps_step,
            max_iter=max_iter, num_random_init=num_random_init)
    elif attack_type == 'bim':
        return BasicIterativeMethod(
            estimator=classifier, eps=epsilon,
            eps_step=eps_step, max_iter=max_iter)
    else:
        raise ValueError(f"Ataque no soportado: {attack_type}")


# ──────────────────────────────────────────────
# Entrenamiento de un modelo desde cero sobre
# un TensorDataset mixto (limpio + adversario)
# ──────────────────────────────────────────────
def train_on_mixed(x_clean, y_clean, x_adv, y_adv,
                   adv_ratio, val_split, device,
                   batch_size=32, epochs=30, lr=1e-4,
                   patience=7, save_path="poisoned_model.pth"):
    """
    Construye un conjunto de entrenamiento mezclando:
        (1 - adv_ratio) * len(x_clean) imágenes limpias
        +
        adv_ratio       * len(x_clean) imágenes adversarias

    Las adversarias sustituyen imágenes del conjunto limpio (no se añaden,
    para mantener el tamaño del dataset constante).

    Entrena un ResNet50 desde pesos ImageNet y devuelve la mejor val bal-acc.
    """
    n_total = len(x_clean)
    n_adv   = int(n_total * adv_ratio)
    n_clean = n_total - n_adv

    # Seleccionamos índices al azar (reproducible)
    rng = np.random.default_rng(SEED)
    adv_idx   = rng.choice(len(x_adv),   size=n_adv,   replace=n_adv > len(x_adv))
    clean_idx = rng.choice(len(x_clean), size=n_clean, replace=False)

    x_mix = np.concatenate([x_clean[clean_idx], x_adv[adv_idx]], axis=0)
    y_mix = np.concatenate([y_clean[clean_idx], y_adv[adv_idx]], axis=0)

    # TensorDataset para el DataLoader
    x_t = torch.from_numpy(x_mix)
    y_t = torch.from_numpy(y_mix).long()
    train_ds = TensorDataset(x_t, y_t)

    # Sampler equilibrado
    counts  = np.bincount(y_mix)
    weights = 1.0 / counts
    sample_w = torch.tensor([weights[y] for y in y_mix], dtype=torch.float)
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)

    # Val loader (limpio, desde HF split)
    val_loader = DataLoader(val_split, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Modelo fresco con pesos ImageNet
    model = build_resnet50(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc    = 0.0
    no_improve      = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(train_ds)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                val_preds.extend(model(xb).argmax(1).cpu().numpy())
                val_labels.extend(yb.numpy())

        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        scheduler.step(val_bal_acc)

        print(f"    Epoch {epoch:3d}/{epochs} | loss {avg_loss:.4f} | "
              f"val bal-acc {val_bal_acc:.4f}")

        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            no_improve   = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping en época {epoch}")
                break

    return best_val_acc, save_path


# ──────────────────────────────────────────────
# Evaluación final en test limpio de A
# ──────────────────────────────────────────────
def evaluate_model(model_path, test_split, device, batch_size=32):
    model = build_resnet50(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(test_split, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.extend(model(xb).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())

    bal_acc   = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall    = recall_score(labels,    preds, average='macro', zero_division=0)
    cm        = confusion_matrix(labels, preds)
    return bal_acc, precision, recall, cm


# ──────────────────────────────────────────────
# Experimento completo
# ──────────────────────────────────────────────
def run_poisoning_experiment(
        data_dir_A, data_dir_X, model_A_path,
        attack_type, epsilon, eps_step, max_iter, num_random_init,
        adv_ratios, batch_size, epochs, lr, patience,
        n_samples_X, output_dir):

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════
    # INSTITUCIÓN A — cargar datos
    # ══════════════════════════════════════════
    print("\n══ INSTITUCIÓN A: cargando datos RIM-ONE ══")
    train_A, val_A, test_A, train_idx_A, base_A = load_institution_data(
        data_dir_A, batch_size)

    # Convertimos train de A a numpy (imágenes limpias para mezclar)
    print("  Extrayendo arrays numpy del train de A...")
    x_train_A, y_train_A = hf_split_to_numpy(train_A, batch_size)
    print(f"  x_train_A: {x_train_A.shape}")

    # ══════════════════════════════════════════
    # INSTITUCIÓN X — generar adversarios
    # ══════════════════════════════════════════
    print("\n══ INSTITUCIÓN X: generando imágenes adversarias con datos Refuge ══")

    # X solo tiene acceso al modelo que A le devolvió entrenado
    model_A = build_resnet50(num_classes=2, pretrained=False).to(device)
    model_A.load_state_dict(torch.load(model_A_path, map_location=device))
    model_A.eval()
    art_classifier = build_art_classifier(model_A, device)

    # Datos de X (Refuge)
    _, _, test_X, _, _ = load_institution_data(data_dir_X, batch_size)
    # X usa su propio conjunto de imágenes para generar adversarios
    # (no conoce los datos de A)
    if n_samples_X:
        test_X = test_X.select(range(min(n_samples_X, len(test_X))))

    print(f"  Extrayendo {len(test_X)} imágenes de X (Refuge)...")
    x_X, y_X = hf_split_to_numpy(test_X, batch_size)

    print(f"  Ejecutando ataque {attack_type.upper()} (ε={epsilon})...")
    attacker = create_attack(attack_type, art_classifier,
                             epsilon, eps_step, max_iter, num_random_init)
    x_X_adv = attacker.generate(x=x_X)

    # Verificar degradación en el modelo de A con sus propias imágenes adversarias
    print("\n  Verificación: accuracy de modelo_A sobre adversarios de X")
    preds_clean = np.argmax(art_classifier.predict(x_X), axis=1)
    preds_adv   = np.argmax(art_classifier.predict(x_X_adv), axis=1)
    print(f"    Bal-Acc limpio   : {balanced_accuracy_score(y_X, preds_clean):.4f}")
    print(f"    Bal-Acc adversario: {balanced_accuracy_score(y_X, preds_adv):.4f}")

    # ══════════════════════════════════════════
    # EXPERIMENTO DE ENVENENAMIENTO
    # A recibe las imágenes adversarias de X y entrena con mezcla creciente
    # ══════════════════════════════════════════
    print("\n══ EXPERIMENTO: entrenamiento con % creciente de adversarios ══")
    print(f"  Ratios a evaluar: {adv_ratios}")

    results = []

    # Línea base: ratio 0.0 (solo imágenes limpias de A)
    for adv_ratio in adv_ratios:
        print(f"\n── Ratio adversario: {adv_ratio:.0%} ──")
        save_path = str(Path(output_dir) / f"model_poisoned_{int(adv_ratio*100):03d}.pth")

        best_val, _ = train_on_mixed(
            x_clean   = x_train_A,
            y_clean   = y_train_A,
            x_adv     = x_X_adv,
            y_adv     = y_X,
            adv_ratio = adv_ratio,
            val_split = val_A,
            device    = device,
            batch_size= batch_size,
            epochs    = epochs,
            lr        = lr,
            patience  = patience,
            save_path = save_path,
        )

        # Evaluación en test LIMPIO de A
        bal_acc, precision, recall, cm = evaluate_model(
            save_path, test_A, device, batch_size)

        print(f"  → Test limpio A | Bal-Acc: {bal_acc:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  Confusion matrix:\n{cm}")

        results.append({
            "adv_ratio" : adv_ratio,
            "val_bal_acc": best_val,
            "test_bal_acc": bal_acc,
            "test_precision": precision,
            "test_recall": recall,
            "confusion_matrix": cm.tolist(),
        })

    # ══════════════════════════════════════════
    # Resumen final
    # ══════════════════════════════════════════
    print("\n══ RESUMEN DEL EXPERIMENTO ══")
    print(f"  {'Ratio':>6}  {'Bal-Acc':>8}  {'Precision':>10}  {'Recall':>8}")
    print(f"  {'------':>6}  {'-------':>8}  {'----------':>10}  {'------':>8}")
    for r in results:
        print(f"  {r['adv_ratio']:>6.0%}  "
              f"{r['test_bal_acc']:>8.4f}  "
              f"{r['test_precision']:>10.4f}  "
              f"{r['test_recall']:>8.4f}")

    # Guardar resultados en JSON
    results_path = Path(output_dir) / "poisoning_results.json"
    # Convertir a tipos serializables
    for r in results:
        r["adv_ratio"] = float(r["adv_ratio"])
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en {results_path}")

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Experimento de envenenamiento con imágenes adversarias")

    # Datos y modelo
    parser.add_argument("--data_dir_A",  type=str, required=True,
                        help="Carpeta de datos de la institución A (RIM-ONE)")
    parser.add_argument("--data_dir_X",  type=str, required=True,
                        help="Carpeta de datos de la institución X (Refuge)")
    parser.add_argument("--model_A",     type=str, required=True,
                        help="Ruta al modelo entrenado de A (best_resnet50.pth)")

    # Ataque
    parser.add_argument("--attack",          type=str,   default="fgsm",
                        choices=["fgsm", "pgd", "bim"])
    parser.add_argument("--epsilon",         type=float, default=0.02)
    parser.add_argument("--eps_step",        type=float, default=None)
    parser.add_argument("--max_iter",        type=int,   default=10)
    parser.add_argument("--num_random_init", type=int,   default=1)

    # Experimento de envenenamiento
    parser.add_argument("--adv_ratios", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Lista de ratios adversarios a evaluar")
    parser.add_argument("--n_samples_X", type=int, default=None,
                        help="Limitar imágenes de X para generar adversarios")

    # Entrenamiento
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=7)

    # Salida
    parser.add_argument("--output_dir", type=str, default="poisoning_outputs",
                        help="Directorio para guardar modelos y resultados")

    args = parser.parse_args()

    run_poisoning_experiment(
        data_dir_A      = args.data_dir_A,
        data_dir_X      = args.data_dir_X,
        model_A_path    = args.model_A,
        attack_type     = args.attack,
        epsilon         = args.epsilon,
        eps_step        = args.eps_step,
        max_iter        = args.max_iter,
        num_random_init = args.num_random_init,
        adv_ratios      = args.adv_ratios,
        batch_size      = args.batch_size,
        epochs          = args.epochs,
        lr              = args.lr,
        patience        = args.patience,
        n_samples_X     = args.n_samples_X,
        output_dir      = args.output_dir,
    )


if __name__ == "__main__":
    main()
