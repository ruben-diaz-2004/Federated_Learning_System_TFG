# -*- coding: utf-8 -*-
"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================

model_training_syft.py
Entrenamiento federado en PySyft para el proyecto de investigación de glaucoma.

Modo de uso:
    python model_training_syft.py --data_dir /ruta/rimone_A --model_out best.pth
"""

import argparse

import numpy as np
import syft as sy
import torch

from database_access import get_experiment

def run_syft_pipeline_db(experiment_id : int) -> dict:
    # Get main experiment parameters
    cfg=get_experiment(experiment_id)



def run_syft_pipeline(
    data_dir:   str,
    model_out:  str  = "best_rimone_syft.pth",
    cfg:        dict = None,
    owner_email:    str = "alu0101552613@ull.edu.es",
    owner_passwd:   str = "glaucoma_research_syft_admin",
    scientist_email:  str = "rachel@datascience.inst",
    scientist_passwd: str = "syftrocks",
) -> dict:
    """
    Ejecuta el pipeline PySyft completo y devuelve el dict de resultados.

    Parámetros
    ----------
    data_dir        : ruta al directorio del dataset en el servidor del owner.
    model_out       : nombre del fichero .pth donde se guardará el modelo.
    cfg             : hiperparámetros de entrenamiento. Valores por defecto:
                        batch_size=32, epochs=500, lr=0.001,
                        patience=50, num_classes=2.
    owner_email/passwd      : credenciales del Data Owner en el servidor Syft.
    scientist_email/passwd  : credenciales del Data Scientist.

    Return
    -------
    dict con las claves:
        status, device, total_imagenes, train_samples, val_samples,
        test_samples, best_epoch, best_val_bal_acc, test_bal_acc,
        test_precision, test_recall, confusion_matrix, model_path, error.
    """

    # ── Valores por defecto de cfg ────────────────────────────────────────────
    _cfg = {
        "batch_size":  32,
        "epochs":      500,
        "lr":          0.001,
        "patience":    50,
        "num_classes": 2,
        "seed":        42,
        "train_ratio": 0.70,
        "val_ratio":   0.10,
    }
    if cfg:
        _cfg.update(cfg)

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 1 — DATA OWNER: setup del servidor y subida del dataset
    # ══════════════════════════════════════════════════════════════════════════
    data_site    = sy.orchestra.launch(name="Glaucoma-research-centre", reset=True)
    owner_client = data_site.login(email="info@openmined.org", password="changethis")

    owner_client.account.set_email(owner_email)
    owner_client.account.set_password(owner_passwd, confirm=False)
    owner_client.account.update(
        name="Ruben, the Data Owner",
        institution="Glaucoma Research Centre",
    )

    owner_client.users.create(
        email=scientist_email, name="Dr. Rachel Science",
        password=scientist_passwd, password_verify=scientist_passwd,
        institution="Data Science Institute",
        website="https://datascience_institute.research.data",
    )

    # Datos reales — solo visibles en el servidor del owner
    raw_data_path = {
        "path":         data_dir,
        "model_out":    model_out,
        "train_config": _cfg,
    }

    # Datos mock — lo que verá el científico
    mock_data = {
        "images":  np.zeros((5, 256, 256, 3), dtype=np.uint8),
        "labels":  [0, 1, 0, 1, 0],
        "train_config": _cfg,
    }

    image_asset     = sy.Asset(name="classification_images", data=raw_data_path, mock=mock_data)
    medical_dataset = sy.Dataset(name="Medical Images Dataset",
        description="Glaucoma research dataset. Labels available: ['Glaucoma', 'Normal']. "
                    "Preprocessing required.",
    )
    medical_dataset.add_asset(image_asset)
    owner_client.upload_dataset(medical_dataset)

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 2 — DATA SCIENTIST: envío de la solicitud de ejecución
    # ══════════════════════════════════════════════════════════════════════════
    data_site = sy.orchestra.launch(name="Glaucoma-research-centre")
    scientist_client = data_site.login(
        email=scientist_email, password=scientist_passwd
    )

    dataset = scientist_client.datasets["Medical Images Dataset"]
    asset   = dataset.assets["classification_images"]

    my_project = sy.Project(
        name="My TFG Project",
        description="Training a ResNet50 model remotely. Only metrics are returned.",
        members=[scientist_client],
    )

    @sy.syft_function(input_policy=sy.ExactMatch(raw_data_dict=asset))
    def remote_preprocessing_and_training(raw_data_dict):
        """
        Pipeline completo ejecutado en el servidor del owner:
          1. Data_Preprocessing  → HuggingFace Dataset
          2. split_dataset       → train / val / test
          3. DataLoaders con WeightedRandomSampler
          4. build_resnet50      → ResNet50 preentrenado en ImageNet
          5. train()             → early stopping + ReduceLROnPlateau
          6. evaluate()          → métricas en test

        Solo se devuelven métricas; el modelo queda guardado en disco
        en el servidor bajo la ruta model_out.
        """
        import traceback
        try:
            import torch
            from pathlib import Path
            from torch.utils.data import DataLoader
            from sklearn.metrics import confusion_matrix

            from data_preprocessing import Data_Preprocessing
            from train_resnet import (
                set_seed, build_resnet50, split_dataset,
                make_balanced_sampler, collate_fn,
                HFTransform, TrainProcessor, ValProcessor,
                train, evaluate,
            )

            data_path   = Path(raw_data_dict["path"])
            model_out   = raw_data_dict.get("model_out", "best_rimone_syft.pth")
            cfg         = raw_data_dict.get("train_config", {})
            batch_size  = cfg.get("batch_size",  32)
            epochs      = cfg.get("epochs",      500)
            lr          = cfg.get("lr",           0.001)
            patience    = cfg.get("patience",     50)
            num_classes = cfg.get("num_classes",  2)
            seed        = cfg.get("seed",         42)
            train_ratio = cfg.get("train_ratio",  0.70)
            val_ratio   = cfg.get("val_ratio",    0.10)

            set_seed(seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 1. Preprocesado
            base_dataset = Data_Preprocessing(
                data_path=data_path, prep_batch_size=batch_size
            ).dataset

            # 2. Split
            train_idx, val_idx, test_idx = split_dataset(base_dataset, train_ratio, val_ratio)
            train_split = base_dataset.select(train_idx)
            val_split   = base_dataset.select(val_idx)
            test_split  = base_dataset.select(test_idx)

            # 3. Transforms
            train_split.set_transform(HFTransform(TrainProcessor()))
            val_split.set_transform(HFTransform(ValProcessor()))
            test_split.set_transform(HFTransform(ValProcessor()))

            # 4. DataLoaders
            sampler = make_balanced_sampler(base_dataset, train_idx)
            train_loader = DataLoader(train_split, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
            test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            # 5. Modelo
            model = build_resnet50(num_classes=num_classes, pretrained=True).to(device)

            # 6. Entrenamiento — train() devuelve (best_val_acc, best_epoch)
            best_val_acc, best_epoch = train(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=lr, patience=patience,
                save_path=model_out,
            )

            # 7. Evaluación en test
            bal_acc, precision, recall = evaluate(model, test_loader, device)

            # 8. Confusion matrix serializable
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
            cm = confusion_matrix(all_labels, all_preds).tolist()

            return {
                "status":           "Exito",
                "device":           str(device),
                "total_imagenes":   len(base_dataset),
                "train_samples":    len(train_idx),
                "val_samples":      len(val_idx),
                "test_samples":     len(test_idx),
                "best_epoch":       int(best_epoch),
                "best_val_bal_acc": round(float(best_val_acc), 4),
                "test_bal_acc":     round(float(bal_acc),      4),
                "test_precision":   round(float(precision),    4),
                "test_recall":      round(float(recall),       4),
                "confusion_matrix": cm,
                "model_path":       model_out,
                "error":            None,
            }

        except Exception:
            return {"status": "CRASH INTERNO", "error": traceback.format_exc()}

    my_project.create_code_request(
        obj=remote_preprocessing_and_training, client=scientist_client
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 3 — DATA OWNER: revisión y aprobación
    # ══════════════════════════════════════════════════════════════════════════
    owner_client = data_site.login(email=owner_email, password=owner_passwd)
    pending_requests = owner_client.requests

    for req in pending_requests:
        print(f"  Request ID: {req.id} | Status: {req.status}")

    incoming_request = pending_requests[0]
    incoming_request.approve()
    print("Solicitud aprobada.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 4 — DATA SCIENTIST: ejecución y descarga de métricas
    # ══════════════════════════════════════════════════════════════════════════
    scientist_client = data_site.login(email=scientist_email, password=scientist_passwd)

    result_pointer = scientist_client.code.remote_preprocessing_and_training(raw_data_dict=asset)
    print("Ejecutando en el servidor remoto...")

    final_result = result_pointer.get()

    print("\nResultados recibidos por el científico:")
    print(f"  Status              : {final_result.get('status')}")
    print(f"  Dispositivo         : {final_result.get('device')}")
    print(f"  Total imágenes      : {final_result.get('total_imagenes')}")
    print(f"  Train / Val / Test  : {final_result.get('train_samples')} / "
          f"{final_result.get('val_samples')} / {final_result.get('test_samples')}")
    print(f"  Mejor época         : {final_result.get('best_epoch')}")
    print(f"  Mejor Val Bal-Acc   : {final_result.get('best_val_bal_acc')}")
    print(f"  Test Bal-Acc        : {final_result.get('test_bal_acc')}")
    print(f"  Test Precision      : {final_result.get('test_precision')}")
    print(f"  Test Recall         : {final_result.get('test_recall')}")
    print(f"  Confusion matrix    : {final_result.get('confusion_matrix')}")
    print(f"  Modelo guardado en  : {final_result.get('model_path')}")

    if final_result.get("error"):
        print(f"\n  ERROR:\n{final_result['error']}")

    data_site.land()
    return final_result


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline PySyft de entrenamiento federado para glaucoma"
    )
    parser.add_argument("--data_dir",   required=True,
                        help="Ruta al directorio del dataset")
    parser.add_argument("--model_out",  default="best_rimone_syft.pth",
                        help="Fichero .pth de salida del modelo")
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--patience",   type=int,   default=50)
    parser.add_argument("--num_classes",type=int,   default=2)
    parser.add_argument("--owner_email",    default="alu0101552613@ull.edu.es")
    parser.add_argument("--owner_passwd",   default="glaucoma_research_syft_admin")
    parser.add_argument("--scientist_email",  default="rachel@datascience.inst")
    parser.add_argument("--scientist_passwd", default="syftrocks")
    return parser.parse_args()


def main():
    args = _parse_args()

    cfg = {
        "batch_size":  args.batch_size,
        "epochs":      args.epochs,
        "lr":          args.lr,
        "patience":    args.patience,
        "num_classes": args.num_classes,
    }

    run_syft_pipeline(
        data_dir         = args.data_dir,
        model_out        = args.model_out,
        cfg              = cfg,
        owner_email      = args.owner_email,
        owner_passwd     = args.owner_passwd,
        scientist_email  = args.scientist_email,
        scientist_passwd = args.scientist_passwd,
    )


if __name__ == "__main__":
    main()
