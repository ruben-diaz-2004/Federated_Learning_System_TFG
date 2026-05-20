# -*- coding: utf-8 -*-
"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================
train_resnet.ipynb

Módulo de entrenamiento de ResNet50 con PyTorch sobre el dataset de imágenes de retina.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from pathlib import Path
from torchvision import transforms, models
from torchvision.transforms.v2 import (
    ColorJitter, RandomCrop, Normalize, ToImage, ToDtype, Compose
)
from data_preprocessing import Data_Preprocessing

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_resnet50(num_classes=2, pretrained=True):
    """
    ResNet50 preentrenado en ImageNet con cabeza personalizada:
        backbone ResNet50 (sin fc original)
        → Dropout(0.2)
        → Flatten
        → Linear(2048, 128) + ReLU
        → Dropout(0.2)
        → Linear(128, num_classes) + Softmax
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet50(weights=weights)

    in_features = backbone.fc.in_features  # 2048
    backbone.fc = nn.Identity()

    model = nn.Sequential(
        backbone,
        nn.Dropout(p=0.2),
        nn.Flatten(),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1),
    )
    return model


# ----------------------------------------------
# Image processors para pasar a Data_Preprocessing
# ----------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class TrainProcessor:
    """
    Processor con augmentación para entrenamiento.
    Las imágenes ya llegan redimensionadas a 256x256 por _prepare.
    Pipeline: RandomCrop(224) → augmentación → ToTensor → Normalize
    Implementado como clase para ser serializable en Windows (multiprocessing spawn).
    """
    def __init__(self):
        self.tf = Compose([
            transforms.CenterCrop(224),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, images):
        return [self.tf(img) for img in images]


class ValProcessor:
    """
    Processor sin augmentación para validación y test.
    Sigue el estándar de PyTorch/ImageNet:
      Resize(256) → CenterCrop(224) → ToTensor → Normalize
    Implementado como clase para ser serializable en Windows (multiprocessing spawn).
    """
    def __init__(self):
        self.tf = Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, images):
        return [self.tf(img) for img in images]


class HFTransform:
    """
    Wrapper serializable para set_transform de HuggingFace.
    Evita el error 'Can't pickle local function' en Windows.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        images = [img.convert("RGB") for img in examples["image"]]
        examples["pixel_values"] = self.processor(images)
        return examples


# ----------------------------------------------
# División train / val / test sobre HF dataset
# ----------------------------------------------
def split_dataset(hf_dataset, train_ratio=0.70, val_ratio=0.10, seed=SEED):
    """
    Divide un HuggingFace dataset en train, val, test de forma reproducible.
    train_ratio : fracción total para train+val  (0.70)
    val_ratio   : fracción del bloque train+val para val (0.10)
    Devuelve: train_idx, val_idx, test_idx  (listas de enteros)
    """
    n = len(hf_dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train_val = int(n * train_ratio)
    train_val_idx = indices[:n_train_val]
    test_idx      = indices[n_train_val:]

    n_val     = int(len(train_val_idx) * val_ratio)
    val_idx   = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]

    print(f"  Total: {n} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


# ----------------------------------------------
# Collate: convierte lista de dicts HF → tensores
# ----------------------------------------------
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels       = [item["label"]        for item in batch]

    if isinstance(pixel_values[0], torch.Tensor):
        images = torch.stack(pixel_values)
    else:
        tf = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        images = torch.stack([tf(pv) for pv in pixel_values])

    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


# ----------------------------------------------
# Sampler equilibrado por clase
# ----------------------------------------------
def make_balanced_sampler(hf_dataset, indices):
    targets = [hf_dataset[i]["label"] for i in indices]
    class_counts  = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ----------------------------------------------
# Métricas
# ----------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    bal_acc   = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels,    all_preds, average='macro', zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)

    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  Precision (macro) : {precision:.4f}")
    print(f"  Recall    (macro) : {recall:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    return bal_acc, precision, recall


# ----------------------------------------------
# Bucle de entrenamiento
# ----------------------------------------------
def train(model, train_loader, val_loader, device,
          epochs=30, lr=1e-4, patience=7, save_path="best_model.pth"):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # -- Entrenamiento --
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # -- Validación --
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                preds  = model(images).argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())

        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        scheduler.step(val_bal_acc)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Val Bal-Acc: {val_bal_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # -- Early stopping + guardado --
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            epochs_no_improve = 0
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Mejor modelo guardado ({save_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping en época {epoch}")
                break

    print(f"\nMejor Val Balanced Accuracy: {best_val_acc:.4f}")
    return best_val_acc, best_epoch

# --------------------------------------------- 
# Training by epochs
# ---------------------------------------------
def train_epochs(model, train_loader, val_loader, device,
          epochs=30, lr=1e-4, save_path="model.pth"):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    val_bal_acc = 0.0
    avg_loss=0.0

    for epoch in range(1, epochs + 1):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                preds  = model(images).argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())

        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        scheduler.step(val_bal_acc)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Val Bal-Acc: {val_bal_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # -- Early stopping + guardado --
        #if val_bal_acc > best_val_acc:
        #    best_val_acc = val_bal_acc
        #    epochs_no_improve = 0
        #    best_epoch = epoch
        #    torch.save(model.state_dict(), save_path)
        #    print(f"  ✓ Mejor modelo guardado ({save_path})")
        #else:
        #    epochs_no_improve += 1
        #    if epochs_no_improve >= patience:
        #        print(f"  Early stopping en época {epoch}")
        #        break

    print(f"\nUltima  Val Balanced Accuracy: {val_bal_acc:.4f}")
    torch.save(model.state_dict(), save_path)

    return val_bal_acc,avg_loss

# ------------------------------------------------
# Recalcular pesos de las etapas de batch normalization
# -------------------------------------------------

def recalculate_norm_weights(model,train_loader,device):
    model.train()
    with torch.no_grad():
        for images,_ in train_loader:
            images=images.to(device)
            model(images)
    model.eval()



def main():

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # -- 1. Cargar dataset base con Data_Preprocessing --
    print("\n-- Cargando dataset con Data_Preprocessing --")
    data_path=Path('C:\\Users\\ruben\\Desktop\\TFG\\workdir\\rimone_A')
    batch_size=32
    preprocessed=Data_Preprocessing(
        data_path=data_path,
        prep_batch_size=batch_size)
    base_dataset = preprocessed.dataset  # HuggingFace Dataset
    test_preprocessed = Data_Preprocessing(
        data_path=data_path,
        prep_batch_size=batch_size,
        split_name='test')
    # -- 2. Dividir índices --
    print("\n-- Dividiendo en train / val / test --")
    train_idx, val_idx, test_idx = split_dataset(base_dataset)

    train_split = base_dataset.select(train_idx)
    val_split   = base_dataset.select(val_idx)
    test_split  = test_preprocessed.dataset

    # -- 3. Asignar processor por split via set_transform --
    train_split.set_transform(HFTransform(TrainProcessor()))
    val_split.set_transform(HFTransform(ValProcessor()))
    test_split.set_transform(HFTransform(ValProcessor()))

    # -- 4. DataLoaders --
    sampler = make_balanced_sampler(base_dataset, train_idx)

    train_loader = DataLoader(train_split, batch_size=batch_size,
                            sampler=sampler, collate_fn=collate_fn,
                            num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_split,   batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=0)
    test_loader  = DataLoader(test_split,  batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=0)

    # -- 5. Modelo --
    print("\n-- Construyendo modelo --")
    model = build_resnet50(num_classes=2, pretrained=True).to(device)

    # -- 6. Entrenamiento --
    patience = 100
    save_path = "best_resnet50.pth"
    epochs = 500
    lr = 0.001
    print("\n-- Entrenamiento --")
    train(model, train_loader, val_loader, device,
        epochs=epochs, lr=lr,
        patience=patience, save_path=save_path)

    # -- 7. Evaluación final en test --
    print("\n-- Evaluación en Test (mejor modelo) --")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
