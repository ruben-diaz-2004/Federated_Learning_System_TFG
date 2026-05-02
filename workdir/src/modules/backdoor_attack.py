"""
backdoor_attack.py
==================
Pipeline de ataque de puerta trasera (PoisoningAttackBackdoor) sobre ResNet50,
con soporte para múltiples tipos de trigger para comparar su efectividad.

Triggers soportados:
  - square        : parche cuadrado blanco relleno
  - cross         : cruz blanca (dos rectángulos perpendiculares)
  - checkerboard  : tablero de ajedrez blanco/negro
  - gaussian      : ruido gaussiano de alta varianza localizado
  - sinusoidal    : bandas sinusoidales horizontales (SIG, global)
  - border        : contorno cuadrado, interior intacto

Escenario: el data scientist malintencionado inyecta el trigger en la función
remota de PySyft. Antes de entrenar, envenena un porcentaje del training set:
toma imágenes de source_class, les aplica el trigger y fuerza su etiqueta
a target_class.

Uso:
    python backdoor_attack.py \
        --data_dir ../../rimone_A \
        --trigger_type square \
        --save_path backdoor_square.pth \
        --save_poisoned ./poisoned_square
"""

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, confusion_matrix
)

# ART — ataque de envenenamiento
from art.attacks.poisoning import PoisoningAttackBackdoor

# Guardado de imágenes
from PIL import Image

from data_preprocessing import Data_Preprocessing
from train_resnet import (
    build_resnet50, split_dataset, make_balanced_sampler, collate_fn,
    HFTransform, TrainProcessor, ValProcessor,
    train, evaluate,
    SEED, set_seed, IMAGENET_MEAN, IMAGENET_STD,
)


# ──────────────────────────────────────────────
# Helpers de shape: centralizan la detección (C,H,W) vs (H,W,C)
# ──────────────────────────────────────────────
def _detect_format(x: np.ndarray) -> tuple[str, int, int]:
    """
    Detecta el formato del array y devuelve ('chw' | 'hwc', H, W).
    Heurística: canales son 1, 3 o 4; alto/ancho son ≥ 32.
    Maneja ndim=3 (una imagen) y ndim=4 (batch).
    """
    if x.ndim == 3:
        if x.shape[0] in (1, 3, 4) and x.shape[1] >= 32:
            return "chw", x.shape[1], x.shape[2]
        return "hwc", x.shape[0], x.shape[1]
    if x.ndim == 4:
        if x.shape[1] in (1, 3, 4) and x.shape[2] >= 32:
            return "chw", x.shape[2], x.shape[3]
        return "hwc", x.shape[1], x.shape[2]
    raise ValueError(f"Array con ndim={x.ndim} no soportado")


def _apply_patch(x: np.ndarray, patch: np.ndarray, r: int, c: int) -> np.ndarray:
    """
    Pega `patch` (H_p, W_p) o (H_p, W_p, C) en x en la posición (r, c).
    Detecta el formato de x y replica el patch a todos los canales si hace falta.
    """
    fmt, _, _ = _detect_format(x)
    h, w = patch.shape[:2]

    if patch.ndim == 2:
        # Patch en escala de grises → replicar a todos los canales
        if fmt == "chw":
            n_ch = x.shape[-3]
            patch_c = np.broadcast_to(patch, (n_ch, h, w))
        else:
            n_ch = x.shape[-1]
            patch_c = np.broadcast_to(patch[..., None], (h, w, n_ch))
    else:
        patch_c = patch

    if x.ndim == 3:
        if fmt == "chw":
            x[:, r:r + h, c:c + w] = patch_c
        else:
            x[r:r + h, c:c + w, :] = patch_c
    else:  # ndim == 4
        if fmt == "chw":
            x[:, :, r:r + h, c:c + w] = patch_c
        else:
            x[:, r:r + h, c:c + w, :] = patch_c
    return x


def _corner_coords(position: str, H: int, W: int, size: int):
    """Devuelve (row_start, col_start) para la esquina indicada."""
    margin = 2
    coords = {
        "top_left":     (margin,            margin),
        "top_right":    (margin,            W - size - margin),
        "bottom_left":  (H - size - margin, margin),
        "bottom_right": (H - size - margin, W - size - margin),
    }
    if position not in coords:
        raise ValueError(f"position debe ser uno de {list(coords.keys())}")
    return coords[position]


# ──────────────────────────────────────────────
# Triggers — todos comparten la firma trigger_fn(x) → x perturbado
# ──────────────────────────────────────────────
def make_square_trigger(size: int = 8, position: str = "top_left"):
    """Parche cuadrado blanco relleno."""
    patch = np.ones((size, size), dtype=np.float32)

    def trigger(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        _, H, W = _detect_format(x)
        r, c = _corner_coords(position, H, W, size)
        return _apply_patch(x, patch, r, c)

    return trigger


def make_cross_trigger(size: int = 8, position: str = "top_left"):
    """
    Cruz blanca: dos rectángulos perpendiculares dentro del recuadro size×size.
    Grosor de los brazos = size // 3 (mínimo 1).
    """
    thickness = max(1, size // 3)
    patch = np.zeros((size, size), dtype=np.float32)
    mid_start = (size - thickness) // 2
    mid_end   = mid_start + thickness
    patch[mid_start:mid_end, :] = 1.0   # brazo horizontal
    patch[:, mid_start:mid_end] = 1.0   # brazo vertical

    def trigger(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        _, H, W = _detect_format(x)
        r, c = _corner_coords(position, H, W, size)
        return _apply_patch(x, patch, r, c)

    return trigger


def make_checkerboard_trigger(size: int = 8, position: str = "top_left",
                              cell_size: int = 2):
    """
    Tablero ajedrez alternando blanco/negro con celdas de cell_size píxeles.
    """
    patch = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                patch[i, j] = 1.0

    def trigger(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        _, H, W = _detect_format(x)
        r, c = _corner_coords(position, H, W, size)
        return _apply_patch(x, patch, r, c)

    return trigger


def make_gaussian_trigger(size: int = 8, position: str = "top_left",
                          mean: float = 0.5, std: float = 0.3):
    """
    Ruido gaussiano localizado en la región size×size.
    El patrón es FIJO (mismo seed) para que el trigger sea consistente
    entre entrenamiento e inferencia.
    """
    rng = np.random.default_rng(SEED)
    patch = rng.normal(loc=mean, scale=std, size=(size, size)).astype(np.float32)
    patch = np.clip(patch, 0.0, 1.0)

    def trigger(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        _, H, W = _detect_format(x)
        r, c = _corner_coords(position, H, W, size)
        return _apply_patch(x, patch, r, c)

    return trigger


def make_sinusoidal_trigger(delta: float = 0.1, frequency: int = 6, **kwargs):
    """
    Trigger SIG (Barni et al. 2019): bandas horizontales sinusoidales
    superpuestas en TODA la imagen. Único trigger global, no localizado.

    pixel(i, j) += delta * sin(2π * frequency * j / W)

    `delta` controla la amplitud (más alto = más visible, más fácil de aprender).
    Acepta **kwargs para compartir firma con los demás (size/position se ignoran).
    """
    def trigger(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        fmt, H, W = _detect_format(x)
        # Patrón sinusoidal a lo ancho de la imagen
        cols = np.arange(W, dtype=np.float32)
        wave = delta * np.sin(2 * np.pi * frequency * cols / W)  # (W,)
        wave_2d = np.broadcast_to(wave, (H, W))                  # (H, W)

        if x.ndim == 3:
            if fmt == "chw":
                x = x + wave_2d[None, :, :]
            else:
                x = x + wave_2d[:, :, None]
        else:  # ndim == 4
            if fmt == "chw":
                x = x + wave_2d[None, None, :, :]
            else:
                x = x + wave_2d[None, :, :, None]

        return np.clip(x, 0.0, 1.0)

    return trigger


def make_border_trigger(size: int = 8, position: str = "top_left",
                        thickness: int = 1):
    """
    Solo el contorno cuadrado en blanco; el interior queda intacto.
    `thickness` es el grosor del borde en píxeles.
    """
    patch = np.zeros((size, size), dtype=np.float32)
    patch[:thickness, :]   = 1.0   # borde superior
    patch[-thickness:, :]  = 1.0   # borde inferior
    patch[:, :thickness]   = 1.0   # borde izquierdo
    patch[:, -thickness:]  = 1.0   # borde derecho

    # `mask` indica qué píxeles del patch sobreescriben x;
    # el interior del patch tiene mask=0 y se preserva la imagen original.
    mask = patch > 0

    def trigger(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        fmt, H, W = _detect_format(x)
        r, c = _corner_coords(position, H, W, size)

        if x.ndim == 3:
            if fmt == "chw":
                region = x[:, r:r + size, c:c + size]
                region[:, mask] = 1.0
            else:
                region = x[r:r + size, c:c + size, :]
                region[mask, :] = 1.0
        else:
            if fmt == "chw":
                region = x[:, :, r:r + size, c:c + size]
                region[:, :, mask] = 1.0
            else:
                region = x[:, r:r + size, c:c + size, :]
                region[:, mask, :] = 1.0

        return x

    return trigger


# ──────────────────────────────────────────────
# Dispatcher: nombre de trigger → función trigger
# ──────────────────────────────────────────────
TRIGGER_REGISTRY = {
    "square":       make_square_trigger,
    "cross":        make_cross_trigger,
    "checkerboard": make_checkerboard_trigger,
    "gaussian":     make_gaussian_trigger,
    "sinusoidal":   make_sinusoidal_trigger,
    "border":       make_border_trigger,
}


def get_trigger(name: str, size: int = 8, position: str = "top_left"):
    """
    Construye la función trigger según el nombre.
    Para 'sinusoidal', size y position se ignoran (es un trigger global).
    """
    if name not in TRIGGER_REGISTRY:
        raise ValueError(f"Trigger '{name}' no soportado. "
                         f"Opciones: {list(TRIGGER_REGISTRY.keys())}")
    factory = TRIGGER_REGISTRY[name]
    if name == "sinusoidal":
        return factory()
    return factory(size=size, position=position)


# ──────────────────────────────────────────────
# Conversión HF dataset → numpy (C, H, W) ∈ [0,1]
# Nota: los tensores ya están normalizados con ImageNet stats
# tras pasar por ValProcessor, por lo que el trigger se aplica
# ANTES de la normalización (sobre píxeles crudos ∈ [0,1]).
# ──────────────────────────────────────────────
def dataset_to_numpy(hf_split, batch_size: int = 32):
    """
    Extrae imágenes y etiquetas de un split HuggingFace.
    Devuelve x (N, C, H, W) float32 y y (N,) int64.
    """
    loader = DataLoader(
        hf_split, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    xs, ys = [], []
    for images, labels in loader:
        xs.append(images.numpy())
        ys.append(labels.numpy())
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.int64),
    )


# ──────────────────────────────────────────────
# Envenenamiento del training set
# ──────────────────────────────────────────────
def poison_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    trigger_fn,
    poison_rate: float = 0.2,
    source_class: int = 0,
    target_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Envenena una fracción `poison_rate` de las muestras de `source_class`.

    Para cada muestra seleccionada:
      1. Aplica trigger_fn  → imagen con parche
      2. Cambia la etiqueta a target_class

    Devuelve (x_poisoned, y_poisoned, n_poisoned).
    El resto del dataset permanece intacto.
    """
    attacker = PoisoningAttackBackdoor(perturbation=trigger_fn)

    source_idx = np.where(y_train == source_class)[0]
    n_poison   = max(1, int(len(source_idx) * poison_rate))
    rng        = np.random.default_rng(SEED)
    chosen_idx = rng.choice(source_idx, size=n_poison, replace=False)

    x_out = x_train.copy()
    y_out = y_train.copy()

    x_poison, y_poison = attacker.poison(
        x_train[chosen_idx],
        y=np.full(n_poison, target_class, dtype=np.int64),
    )
    x_out[chosen_idx] = x_poison
    y_out[chosen_idx] = y_poison.astype(np.int64)

    print(f"  Muestras de clase {source_class}: {len(source_idx)} | "
          f"Envenenadas: {n_poison} ({poison_rate*100:.1f}%)")
    return x_out, y_out, n_poison, chosen_idx


# ──────────────────────────────────────────────
# Dataset PyTorch desde arrays numpy
# Permite usar el bucle de entrenamiento existente
# sin modificar train_resnet.py
# ──────────────────────────────────────────────
class NumpyDataset(torch.utils.data.Dataset):
    """Dataset PyTorch construido desde arrays numpy (N, C, H, W)."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_numpy_loader(x, y, batch_size, shuffle=False, balanced=False):
    """
    Crea un DataLoader desde arrays numpy.
    Si balanced=True aplica WeightedRandomSampler para equilibrar clases.
    """
    ds = NumpyDataset(x, y)
    if balanced:
        class_counts  = np.bincount(y)
        class_weights = 1.0 / class_counts
        sample_w      = [class_weights[label] for label in y]
        sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ──────────────────────────────────────────────
# Evaluación de métricas
# ──────────────────────────────────────────────
def evaluate_loader(model, loader, device, label=""):
    """Evalúa el modelo sobre un DataLoader, imprime y devuelve métricas."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    bal_acc   = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels,    all_preds, average='macro', zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)

    print(f"  [{label}] Balanced Accuracy: {bal_acc:.4f} | "
          f"Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    return bal_acc, precision, recall


def compute_attack_success_rate(
    model, x_test: np.ndarray, y_test: np.ndarray,
    trigger_fn, source_class: int, target_class: int,
    batch_size: int, device,
) -> float:
    """
    Attack Success Rate (ASR): fracción de muestras de source_class que,
    al recibir el trigger, se clasifican como target_class.

    Solo se evalúan las muestras de source_class para que la métrica sea pura
    (no contaminada por muestras que ya pertenecían a target_class).
    """
    source_mask = y_test == source_class
    x_source    = x_test[source_mask]

    if len(x_source) == 0:
        print("  ⚠ No hay muestras de source_class en el test set.")
        return 0.0

    x_triggered = trigger_fn(x_source)
    loader      = make_numpy_loader(
        x_triggered,
        np.full(len(x_triggered), source_class, dtype=np.int64),  # etiqueta real, no importa
        batch_size=batch_size,
    )

    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().numpy())

    asr = np.mean(np.array(preds) == target_class)
    print(f"  Attack Success Rate: {asr:.4f}  "
          f"({int(asr * len(x_source))}/{len(x_source)} muestras → clase {target_class})")
    return float(asr)


# ──────────────────────────────────────────────
# Función remota de Syft (data scientist malicioso)
# Se registra como syft_function y se ejecuta en el servidor del owner.
# El envenenamiento ocurre dentro de esta función, invisible para el owner
# hasta que inspecciona el código antes de aprobar la solicitud.
# ──────────────────────────────────────────────
def build_backdoor_syft_function(asset, scientist_client, poison_rate, trigger_size, target_class,
                                  trigger_type="square"):
    """
    Construye y registra la syft_function maliciosa.
    Llama a esta función desde la sección del data scientist.
    """
    import syft as sy

    @sy.syft_function(input_policy=sy.ExactMatch(raw_data_dict=asset))
    def remote_backdoor_training(raw_data_dict):
        """
        Pipeline de entrenamiento con backdoor inyectado por el data scientist.
        El owner solo ve métricas; el modelo envenenado queda en su servidor.
        """
        import traceback
        try:
            import torch
            import numpy as np
            from pathlib import Path
            from torch.utils.data import DataLoader

            from data_preprocessing import Data_Preprocessing
            from train_resnet import (
                set_seed, build_resnet50, split_dataset,
                HFTransform, ValProcessor, TrainProcessor,
                train, SEED,
            )
            from backdoor_attack import (
                get_trigger, dataset_to_numpy,
                poison_dataset, make_numpy_loader,
                evaluate_loader, compute_attack_success_rate,
            )

            # ── Parámetros ──
            data_path    = Path(raw_data_dict["path"])
            cfg          = raw_data_dict.get("train_config", {})
            batch_size   = cfg.get("batch_size",  32)
            epochs       = cfg.get("epochs",      500)
            lr           = cfg.get("lr",           0.001)
            patience     = cfg.get("patience",     50)
            num_classes  = cfg.get("num_classes",  2)

            # Parámetros del backdoor (hardcodeados por el atacante)
            _poison_rate   = poison_rate    # capturado del closure
            _trigger_size  = trigger_size
            _trigger_type  = trigger_type
            _source_class  = 0
            _target_class  = target_class

            set_seed(SEED)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ── 1. Dataset y split ──
            base_dataset = Data_Preprocessing(
                data_path=data_path, prep_batch_size=batch_size
            ).dataset
            train_idx, val_idx, test_idx = split_dataset(base_dataset)

            train_split = base_dataset.select(train_idx)
            val_split   = base_dataset.select(val_idx)
            test_split  = base_dataset.select(test_idx)

            train_split.set_transform(HFTransform(TrainProcessor()))
            val_split.set_transform(HFTransform(ValProcessor()))
            test_split.set_transform(HFTransform(ValProcessor()))

            # ── 2. Convertir a numpy para envenenar ──
            from backdoor_attack import dataset_to_numpy, make_numpy_loader
            x_train, y_train = dataset_to_numpy(train_split, batch_size)
            x_val,   y_val   = dataset_to_numpy(val_split,   batch_size)
            x_test,  y_test  = dataset_to_numpy(test_split,  batch_size)

            # ── 3. Envenenamiento del training set ──
            trigger_fn = get_trigger(_trigger_type, size=_trigger_size, position="top_left")
            x_train_p, y_train_p, n_poisoned, _ = poison_dataset(
                x_train, y_train,
                trigger_fn=trigger_fn,
                poison_rate=_poison_rate,
                source_class=_source_class,
                target_class=_target_class,
            )

            train_loader = make_numpy_loader(x_train_p, y_train_p, batch_size, balanced=True)
            val_loader   = make_numpy_loader(x_val,     y_val,     batch_size)
            test_loader  = make_numpy_loader(x_test,    y_test,    batch_size)

            # ── 4. Entrenamiento ──
            model = build_resnet50(num_classes=num_classes, pretrained=True).to(device)
            best_val_acc = train(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=lr, patience=patience,
                save_path="backdoor_resnet50.pth",
            )

            # ── 5. Evaluación ──
            clean_bal, clean_prec, clean_rec = evaluate_loader(
                model, test_loader, device, label="limpio"
            )
            asr = compute_attack_success_rate(
                model, x_test, y_test,
                trigger_fn=trigger_fn,
                source_class=_source_class,
                target_class=_target_class,
                batch_size=batch_size,
                device=device,
            )

            return {
                "status":             "Exito",
                "device":             str(device),
                "n_poisoned":         n_poisoned,
                "poison_rate":        _poison_rate,
                "trigger_size":       _trigger_size,
                "target_class":       _target_class,
                "best_val_bal_acc":   round(float(best_val_acc[0]), 4),
                "clean_bal_acc":      round(float(clean_bal),    4),
                "clean_precision":    round(float(clean_prec),   4),
                "clean_recall":       round(float(clean_rec),    4),
                "attack_success_rate": round(float(asr),         4),
                "model_weights":      {k: v.cpu() for k, v in model.state_dict().items()},
                "error":              None,
            }

        except Exception:
            import traceback
            return {"status": "CRASH INTERNO", "error": traceback.format_exc()}

    scientist_client.code.request_code_execution(remote_backdoor_training)
    print("Solicitud de backdoor enviada. Esperando aprobación del owner...")
    return remote_backdoor_training


# ──────────────────────────────────────────────
# Guardado de imágenes envenenadas como PNG
# ──────────────────────────────────────────────
def save_poisoned_as_png(
    x_poisoned: np.ndarray,
    y_poisoned: np.ndarray,
    chosen_idx: np.ndarray,
    out_dir: str,
    target_class: int,
) -> None:
    """
    Guarda únicamente las imágenes que recibieron el trigger como PNG.

    chosen_idx: índices exactos devueltos por poison_dataset, evitando
    cualquier ambigüedad sobre qué muestras fueron envenenadas.

    Las imágenes llegan normalizadas con stats ImageNet (C, H, W) float32,
    por lo que se desnormalizan antes de guardar para que el PNG sea visualmente
    correcto.

    Desnormalización: pixel = (tensor * std) + mean  → clip a [0, 1] → uint8
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)

    out_path = Path(out_dir) / f"class_{target_class}"
    out_path.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(chosen_idx):
        img_chw = x_poisoned[idx] * std + mean
        img_chw = np.clip(img_chw, 0.0, 1.0)
        img_hwc = (img_chw.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img_hwc).save(out_path / f"poisoned_{i:04d}.png")

    print(f"  {len(chosen_idx)} imágenes envenenadas guardadas en {out_path}")


# ──────────────────────────────────────────────
# Pipeline principal (ejecución local, sin Syft)
# Útil para experimentos rápidos y desarrollo
# ──────────────────────────────────────────────
def run_backdoor(
    data_dir:      str,
    trigger_type:  str   = "square",
    save_path:     str   = "backdoor_resnet50.pth",
    poison_rate:   float = 0.2,
    trigger_size:  int   = 8,
    trigger_pos:   str   = "top_left",
    source_class:  int   = 0,
    target_class:  int   = 1,
    batch_size:    int   = 32,
    epochs:        int   = 500,
    lr:            float = 0.001,
    patience:      int   = 50,
    save_poisoned: str   = None,
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── 1. Dataset y split ──
    print("\n── Cargando dataset ──")
    base_dataset = Data_Preprocessing(
        data_path=Path(data_dir), prep_batch_size=batch_size
    ).dataset
    train_idx, val_idx, test_idx = split_dataset(base_dataset)
    print(f"  Total: {len(base_dataset)} | "
          f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_split = base_dataset.select(train_idx)
    val_split   = base_dataset.select(val_idx)
    test_split  = base_dataset.select(test_idx)

    train_split.set_transform(HFTransform(TrainProcessor()))
    val_split.set_transform(HFTransform(ValProcessor()))
    test_split.set_transform(HFTransform(ValProcessor()))

    # ── 2. Convertir a numpy ──
    print("\n── Extrayendo arrays numpy ──")
    x_train, y_train = dataset_to_numpy(train_split, batch_size)
    x_val,   y_val   = dataset_to_numpy(val_split,   batch_size)
    x_test,  y_test  = dataset_to_numpy(test_split,  batch_size)
    print(f"  x_train: {x_train.shape} | x_val: {x_val.shape} | x_test: {x_test.shape}")

    # ── 3. Trigger y envenenamiento ──
    print(f"\n── Envenenamiento [trigger={trigger_type}] "
          f"(poison_rate={poison_rate}, size={trigger_size}px, pos={trigger_pos}) ──")
    trigger_fn = get_trigger(trigger_type, size=trigger_size, position=trigger_pos)

    x_train_p, y_train_p, n_poisoned, chosen_idx = poison_dataset(
        x_train, y_train,
        trigger_fn=trigger_fn,
        poison_rate=poison_rate,
        source_class=source_class,
        target_class=target_class,
    )

    if save_poisoned:
        # Si el usuario no especifica ruta exacta, anexamos el tipo de trigger
        # para no pisar resultados de comparativas previas
        out_dir = save_poisoned
        if Path(out_dir).name == "poisoned_images":
            out_dir = f"{out_dir}_{trigger_type}"
        print(f"\n── Guardando imágenes envenenadas en {out_dir} ──")
        save_poisoned_as_png(x_train_p, y_train_p, chosen_idx, out_dir, target_class)

    # ── 4. DataLoaders desde numpy ──
    train_loader = make_numpy_loader(x_train_p, y_train_p, batch_size, balanced=True)
    val_loader   = make_numpy_loader(x_val,     y_val,     batch_size)
    test_loader  = make_numpy_loader(x_test,    y_test,    batch_size)

    # ── 5. Modelo y entrenamiento ──
    print("\n── Construyendo modelo ──")
    model = build_resnet50(num_classes=2, pretrained=True).to(device)

    print("\n── Entrenamiento sobre dataset envenenado ──")
    best_val_acc = train(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, patience=patience, save_path=save_path,
    )

    # Cargar mejor checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device))

    # ── 6. Evaluación sobre test limpio ──
    print("\n── Evaluación sobre test limpio ──")
    clean_bal, clean_prec, clean_rec = evaluate_loader(
        model, test_loader, device, label="limpio"
    )

    # ── 7. Attack Success Rate ──
    print("\n── Attack Success Rate (test con trigger) ──")
    asr = compute_attack_success_rate(
        model, x_test, y_test,
        trigger_fn=trigger_fn,
        source_class=source_class,
        target_class=target_class,
        batch_size=batch_size,
        device=device,
    )

    # ── Resumen ──
    print("\n══ Resumen ══")
    print(f"  Tipo de trigger        : {trigger_type}")
    print(f"  Muestras envenenadas   : {n_poisoned}")
    print(f"  Best val bal-acc       : {best_val_acc[0]:.4f}")
    print(f"  Test bal-acc (limpio)  : {clean_bal:.4f}")
    print(f"  Test precision         : {clean_prec:.4f}")
    print(f"  Test recall            : {clean_rec:.4f}")
    print(f"  Attack Success Rate    : {asr:.4f}")

    return {
        "trigger_type":        trigger_type,
        "n_poisoned":          n_poisoned,
        "best_val_bal_acc":    best_val_acc[0],
        "clean_bal_acc":       clean_bal,
        "clean_precision":     clean_prec,
        "clean_recall":        clean_rec,
        "attack_success_rate": asr,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backdoor attack (PoisoningAttackBackdoor) sobre ResNet50 - Glaucoma. "
                    "Soporta múltiples triggers para análisis comparativo."
    )
    parser.add_argument("--data_dir",      type=str,   required=True,
                        help="Ruta al dataset (rimone_A o refuge_x)")
    parser.add_argument("--trigger_type",  type=str,   default="square",
                        choices=list(TRIGGER_REGISTRY.keys()),
                        help="Tipo de trigger a aplicar")
    parser.add_argument("--save_path",     type=str,   default="backdoor_resnet50.pth",
                        help="Ruta donde guardar el modelo envenenado")
    parser.add_argument("--poison_rate",   type=float, default=0.2,
                        help="Fracción de muestras de source_class a envenenar (0-1)")
    parser.add_argument("--trigger_size",  type=int,   default=8,
                        help="Lado del trigger en píxeles (ignorado en sinusoidal)")
    parser.add_argument("--trigger_pos",   type=str,   default="top_left",
                        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
                        help="Posición del trigger (ignorado en sinusoidal)")
    parser.add_argument("--source_class",  type=int,   default=1,
                        help="Clase origen: la que recibe el trigger en inferencia")
    parser.add_argument("--target_class",  type=int,   default=0,
                        help="Clase objetivo: a la que redirige el trigger")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--epochs",        type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--patience",      type=int,   default=50)
    parser.add_argument("--save_poisoned", type=str,   default=None,
                        help="Directorio donde guardar las imágenes envenenadas como PNG")
    args = parser.parse_args()

    run_backdoor(
        data_dir      = args.data_dir,
        trigger_type  = args.trigger_type,
        save_path     = args.save_path,
        poison_rate   = args.poison_rate,
        trigger_size  = args.trigger_size,
        trigger_pos   = args.trigger_pos,
        source_class  = args.source_class,
        target_class  = args.target_class,
        batch_size    = args.batch_size,
        epochs        = args.epochs,
        lr            = args.lr,
        patience      = args.patience,
        save_poisoned = args.save_poisoned,
    )


if __name__ == "__main__":
    main()
