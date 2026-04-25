"""
Paso 3: Ataques de adversarios sobre ResNet50 con ART (Adversarial Robustness Toolbox)
Adaptado del notebook de referencia (Keras/TF → PyTorch)

Ataques soportados:
  - FGSM  : Fast Gradient Sign Method
  - PGD   : Projected Gradient Descent
  - BIM   : Basic Iterative Method

Uso:
    python adversarial_attacks.py \
        --model_path best_resnet50.pth \
        --data_dir /ruta/rimone \
        --attack fgsm
        python .\adversarial_attacks.py --model_path .\best_resnet50.pth --data_dir ..\..\rimone_A\ --attack fgsm
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.transforms.v2 import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, \
    ColorJitter, Normalize, ToTensor, Compose
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# ART - PyTorch
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod

from data_preprocessing import Data_Preprocessing
# Reutilizamos las funciones del script de entrenamiento
from train_resnet_final import (
    build_resnet50, split_dataset, make_balanced_sampler, collate_fn,
    TrainProcessor, ValProcessor, HFTransform, SEED, set_seed,
    IMAGENET_MEAN, IMAGENET_STD
)

# ──────────────────────────────────────────────
# Wrapper ART para PyTorch
# ──────────────────────────────────────────────
def build_art_classifier(model, device, num_classes=2):
    """
    Envuelve el modelo PyTorch en un PyTorchClassifier de ART.

    ART trabaja internamente con numpy arrays en formato (N, C, H, W) y valores [0, 1].
    La normalización ImageNet ya está dentro del pipeline de transforms, por lo que
    indicamos clip_values=(0, 1) sobre los valores pre-normalización.

    Se usa CrossEntropyLoss porque el modelo incluye Softmax; ART necesita el loss
    para calcular gradientes en los ataques basados en gradiente.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),   # (C, H, W) — formato PyTorch
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),      # rango de los píxeles antes de normalizar
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier


# ──────────────────────────────────────────────
# Creación del atacante
# ──────────────────────────────────────────────
def create_attack(attack_type, classifier,
                  epsilon=0.02, eps_step=None, max_iter=10, num_random_init=1):
    """
    Crea el objeto de ataque de ART según el tipo especificado.

    epsilon        : perturbación máxima (en escala [0,1])
    eps_step       : paso por iteración (por defecto epsilon/4)
    max_iter       : iteraciones máximas para PGD y BIM
    num_random_init: inicializaciones aleatorias para PGD
    """
    if eps_step is None:
        eps_step = epsilon / 4.0

    if attack_type == 'fgsm':
        # Un solo paso en la dirección del gradiente del loss
        attacker = FastGradientMethod(
            estimator=classifier,
            eps=epsilon,
        )
    elif attack_type == 'pgd':
        # Iterativo con proyección en la bola L∞(epsilon)
        attacker = ProjectedGradientDescent(
            estimator=classifier,
            eps=epsilon,
            eps_step=eps_step,
            max_iter=max_iter,
            num_random_init=num_random_init,
        )
    elif attack_type == 'bim':
        # Iterativo sin re-inicialización aleatoria (caso base de PGD)
        attacker = BasicIterativeMethod(
            estimator=classifier,
            eps=epsilon,
            eps_step=eps_step,
            max_iter=max_iter,
        )
    else:
        raise ValueError(f"Ataque '{attack_type}' no soportado. Elige: fgsm, pgd, bim")

    return attacker


# ──────────────────────────────────────────────
# Conversión dataset HF → numpy para ART
# ──────────────────────────────────────────────
def dataset_to_numpy(hf_split, batch_size=32):
    """
    Extrae todos los tensores del split de HuggingFace y los devuelve como
    numpy arrays (N, C, H, W) en float32.
    ART acepta (N, C, H, W) para modelos PyTorch.
    """
    loader = DataLoader(hf_split, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    all_images, all_labels = [], []
    for images, labels in loader:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    x = np.concatenate(all_images, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)
    return x, y


# ──────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────
def evaluate_numpy(classifier, x, y, label=""):
    """
    Evalúa el clasificador ART sobre arrays numpy.
    classifier.predict devuelve probabilidades (N, num_classes).
    """
    preds = np.argmax(classifier.predict(x), axis=1)
    bal_acc   = balanced_accuracy_score(y, preds)
    precision = precision_score(y, preds, average='macro', zero_division=0)
    recall    = recall_score(y,    preds, average='macro', zero_division=0)
    cm        = confusion_matrix(y, preds)
    print(f"  [{label}] Balanced Accuracy: {bal_acc:.4f} | "
          f"Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    return bal_acc, precision, recall


# ──────────────────────────────────────────────
# Visualización de ejemplos adversarios
# ──────────────────────────────────────────────
def visualize_adversarial(x_clean, x_adv, y_true, classifier, 
                           attack_type, n_samples=5, epsilon=0.2, save_path=None):
    """
    Muestra n_samples imágenes limpias vs adversarias con sus predicciones.
    """
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)

    def denormalize(img):
        """Desnormaliza ImageNet y convierte a HWC para matplotlib."""
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img.transpose(1, 2, 0)  # CHW → HWC

    preds_clean = np.argmax(classifier.predict(x_clean[:n_samples]), axis=1)
    preds_adv   = np.argmax(classifier.predict(x_adv[:n_samples]),   axis=1)
    perturbations = x_adv[:n_samples] - x_clean[:n_samples]

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", f"Adversaria ({attack_type.upper()})"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold')

    class_names = {0: "No Glaucoma", 1: "Glaucoma"}

    for i in range(n_samples):
        img_clean = denormalize(x_clean[i])
        img_adv   = denormalize(x_adv[i])
        #img_pert  = np.clip(perturbations[i].transpose(1, 2, 0) * 10 + 0.5, 0, 1)

        true_label  = class_names[y_true[i]]
        pred_clean  = class_names[preds_clean[i]]
        pred_adv    = class_names[preds_adv[i]]

        axes[i, 0].imshow(img_clean)
        axes[i, 0].set_xlabel(f"Real: {true_label}\nPred: {pred_clean}", fontsize=10)

        axes[i, 1].imshow(img_adv)
        color = "red" if preds_adv[i] != y_true[i] else "green"
        axes[i, 1].set_xlabel(f"Pred: {pred_adv}", fontsize=10, color=color)

        #axes[i, 2].imshow(img_pert)
        #linf = np.max(np.abs(perturbations[i]))
        #axes[i, 2].set_xlabel(f"L∞ = {linf:.4f}", fontsize=10)

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(f"Ataque {attack_type.upper()} — ε={epsilon}", fontsize=15, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Figura guardada en {save_path}")
    plt.show()

# ──────────────────────────────────────────────
# Experimento principal
# ──────────────────────────────────────────────
def run_attack(data_dir, model_path, attack_type,
               epsilon, eps_step, max_iter, num_random_init,
               batch_size, n_samples, save_adv):

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── 1. Cargar y dividir dataset ──
    print("\n── Cargando dataset ──")
    data_prep = Data_Preprocessing(
        data_path=Path(data_dir),
        split_name='test',
        image_size=[256, 256],
        image_processor=None,
        num_proc=1,
        prep_batch_size=batch_size,
    )
    # base_dataset = data_prep.dataset
    # train_idx, val_idx, test_idx = split_dataset(base_dataset)

    # Usamos el split de test con ValProcessor (sin augmentación)
    # test_split = base_dataset.select(test_idx)
    test_split = data_prep.dataset
    test_split.set_transform(HFTransform(ValProcessor()))

    # Opcionalmente limitamos el número de muestras para no quedarnos sin memoria
    # if n_samples and n_samples < len(test_idx):
        # test_split = test_split.select(range(n_samples))
        # print(f"  Usando {n_samples} muestras del conjunto de test")

    # ── 2. Convertir a numpy ──
    print("\n── Extrayendo arrays numpy del dataset ──")
    x_test, y_test = dataset_to_numpy(test_split, batch_size)
    print(f"  x_test: {x_test.shape}  y_test: {y_test.shape}")

    # ── 3. Cargar modelo y crear clasificador ART ──
    print("\n── Cargando modelo ──")
    model = build_resnet50(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    classifier = build_art_classifier(model, device)

    # ── 4. Evaluar sobre ejemplos limpios ──
    print("\n── Evaluación sobre ejemplos limpios ──")
    evaluate_numpy(classifier, x_test, y_test, label="limpio")

    # ── 5. Generar ejemplos adversarios ──
    print(f"\n── Generando ejemplos adversarios ({attack_type.upper()}) ──")
    print(f"  ε={epsilon}, eps_step={eps_step or epsilon/4:.4f}, "
          f"max_iter={max_iter}, num_random_init={num_random_init}")

    attacker = create_attack(attack_type, classifier,
                             epsilon=epsilon,
                             eps_step=eps_step,
                             max_iter=max_iter,
                             num_random_init=num_random_init)

    x_test_adv = attacker.generate(x=x_test)
    print(f"  Perturbación media (L∞): {np.max(np.abs(x_test_adv - x_test)):.4f}")

    # ── 6. Evaluar sobre ejemplos adversarios ──
    print("\n── Evaluación sobre ejemplos adversarios ──")
    evaluate_numpy(classifier, x_test_adv, y_test, label="adversario")

    # ── 7. Visualizar ejemplos ──
    print("\n── Visualizando ejemplos antes/después del ataque ──")
    visualize_adversarial(
        x_clean    = x_test,
        x_adv      = x_test_adv,
        y_true     = y_test,
        classifier = classifier,
        attack_type = attack_type,
        n_samples  = 5,
        epsilon = epsilon,
        save_path  = f"adversarial_examples_{attack_type}.png"
    )

    # ── 7. Guardar ejemplos adversarios (opcional) ──
    if save_adv:
        out_path = Path(save_adv)
        np.save(out_path / f"x_adv_{attack_type}.npy", x_test_adv)
        np.save(out_path / f"y_adv_{attack_type}.npy", y_test)
        print(f"\n  Ejemplos adversarios guardados en {save_adv}")

    return x_test, x_test_adv, y_test


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ataques adversarios sobre ResNet50 - Glaucoma")
    parser.add_argument("--data_dir",        type=str, required=True)
    parser.add_argument("--model_path",      type=str, required=True,
                        help="Ruta al .pth del modelo entrenado")
    parser.add_argument("--attack",          type=str, default="fgsm",
                        choices=["fgsm", "pgd", "bim"])
    parser.add_argument("--epsilon",         type=float, default=0.2,
                        help="Perturbación máxima L∞")
    parser.add_argument("--eps_step",        type=float, default=None,
                        help="Paso por iteración (default: epsilon/4)")
    parser.add_argument("--max_iter",        type=int,   default=10)
    parser.add_argument("--num_random_init", type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--n_samples",       type=int,   default=None,
                        help="Limitar número de muestras del test (None = todas)")
    parser.add_argument("--save_adv",        type=str,   default=None,
                        help="Directorio donde guardar los arrays adversarios (.npy)")
    args = parser.parse_args()

    run_attack(
        data_dir        = args.data_dir,
        model_path      = args.model_path,
        attack_type     = args.attack,
        epsilon         = args.epsilon,
        eps_step        = args.eps_step,
        max_iter        = args.max_iter,
        num_random_init = args.num_random_init,
        batch_size      = args.batch_size,
        n_samples       = args.n_samples,
        save_adv        = args.save_adv,
    )


if __name__ == "__main__":
    main()
