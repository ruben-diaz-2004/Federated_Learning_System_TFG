# create_dataset.py
# ─────────────────────────────────────────────────────────────
# Extrae imágenes de archivos .zip a un único directorio con
# subdirectorios por clase, y registra el dataset en la BD.
#
# Estructura resultante:
#   <dataset_dir>/
#       normal/
#           img001.jpg
#           img002.jpg
#           ...
#       glaucoma/
#           img101.jpg
#           ...
#
# Uso:
#   python create_dataset.py \
#       --dataset_dir  ../../refuge_x            \
#       --dataset_name REFUGE                    \
#       --zips         normal.zip glaucoma.zip   \
#       --classes      normal     glaucoma
# ─────────────────────────────────────────────────────────────

import argparse
import zipfile as zf
from pathlib import Path

import numpy as np

from db_1_1 import register_dataset


# ─────────────────────────────────────────────────────────────
# Extracción
# ─────────────────────────────────────────────────────────────

def extract_zips_to_dataset(
    dataset_dir: Path,
    zip_paths: list[Path],
    class_names: list[str],
) -> dict[str, int]:
    """
    Extrae cada zip en su subdirectorio de clase correspondiente.
    zip_paths y class_names deben ir en el mismo orden.

    Retorna un dict {class_name: n_samples}.
    """
    if len(zip_paths) != len(class_names):
        raise ValueError(
            f"Número de zips ({len(zip_paths)}) y clases ({len(class_names)}) no coincide."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for zip_path, class_name in zip(zip_paths, class_names):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)

        print(f"\n── Extrayendo '{zip_path.name}' → '{class_dir}' ──")
        with zf.ZipFile(zip_path, "r") as zfile:
            entries = [e for e in zfile.namelist() if not e.endswith("/")]
            extracted = 0
            for entry in entries:
                file_name = Path(entry).name
                dest = class_dir / file_name
                if dest.exists():
                    print(f"   [SKIP] Ya existe: {file_name}")
                    continue
                # Extraer con nombre limpio (sin rutas internas del zip)
                info = zfile.getinfo(entry)
                info.filename = file_name
                zfile.extract(entry, path=class_dir)
                extracted += 1

        n_files = len(list(class_dir.iterdir()))
        counts[class_name] = n_files
        print(f"   Extraídos: {extracted}  |  Total en directorio: {n_files}")

    return counts


# ─────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────

def create_dataset(
    dataset_dir: Path,
    dataset_name: str,
    zip_paths: list[Path],
    class_names: list[str],
) -> dict:
    """
    1. Extrae los zips al directorio del dataset.
    2. Cuenta las muestras por clase.
    3. Registra el dataset en la BD (idempotente por nombre).

    Retorna dict con dataset_id y conteos.
    """
    # ── 1. Extracción ──────────────────────────────────────────
    counts = extract_zips_to_dataset(dataset_dir, zip_paths, class_names)

    total      = sum(counts.values())
    n_class0   = counts.get(class_names[0], 0)
    n_class1   = counts.get(class_names[1], 0) if len(class_names) > 1 else 0

    print(f"\n── Resumen del dataset ──")
    for cls, n in counts.items():
        print(f"   {cls}: {n}")
    print(f"   Total: {total}")

    # ── 2. Registro en BD ──────────────────────────────────────
    dataset_id = register_dataset(
        name           = dataset_name,
        path           = str(dataset_dir),
        total_samples  = total,
        samples_class0 = n_class0,
        samples_class1 = n_class1,
    )
    print(f"\n── Dataset registrado en BD ──")
    print(f"   dataset_id = {dataset_id}  (nombre='{dataset_name}')")
    print(f"\n   Usa este dataset_id en register_split.py cuando vayas a entrenar.")

    return {
        "dataset_id": dataset_id,
        "total":      total,
        "counts":     counts,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crea el directorio del dataset desde zips y lo registra en la BD."
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Directorio de destino donde se crearán los subdirectorios por clase."
    )
    parser.add_argument(
        "--dataset_name", required=True,
        help="Nombre identificador del dataset en la BD (p.ej. REFUGE)."
    )
    parser.add_argument(
        "--zips", nargs="+", required=True,
        help="Rutas a los archivos .zip, en el mismo orden que --classes."
    )
    parser.add_argument(
        "--classes", nargs="+", required=True,
        help="Nombres de clase, en el mismo orden que --zips (p.ej. normal glaucoma)."
    )
    args = parser.parse_args()

    create_dataset(
        dataset_dir  = Path(args.dataset_dir),
        dataset_name = args.dataset_name,
        zip_paths    = [Path(z) for z in args.zips],
        class_names  = args.classes,
    )


if __name__ == "__main__":
    main()
