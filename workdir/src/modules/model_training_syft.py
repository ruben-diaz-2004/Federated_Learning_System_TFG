# -*- coding: utf-8 -*-
"""
syft_glaucoma.py
================
Pipeline completo PySyft para el proyecto de investigación de glaucoma.
...
"""

import numpy as np
import syft as sy
import torch

print(sy.__version__)


def main():
    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 1 — DATA OWNER: setup del servidor y subida del dataset
    # ══════════════════════════════════════════════════════════════════════════════

    data_site = sy.orchestra.launch(name="Glaucoma-research-centre", reset=True)
    owner_client = data_site.login(email="info@openmined.org", password="changethis")

    OWNER_EMAIL  = "alu0101552613@ull.edu.es"
    OWNER_PASSWD = "glaucoma_research_syft_admin"

    owner_client.account.set_email(OWNER_EMAIL)
    owner_client.account.set_password(OWNER_PASSWD, confirm=False)
    owner_client.account.update(name="Ruben, the Data Owner", institution="Glaucoma Research Centre")

    owner_client.users.create(
        email="rachel@datascience.inst", name="Dr. Rachel Science",
        password="syftrocks", password_verify="syftrocks",
        institution="Data Science Institute",
        website="https://datascience_institute.research.data"
    )

    # Datos reales — solo visibles en el servidor del owner
    raw_data_path = {
        "path": "C:\\Users\\ruben\\Desktop\\TFG\\workdir\\rimone_A",
        "train_config": {
            "batch_size": 32, "epochs": 500, "lr": 0.001,
            "patience": 10, "num_classes": 2
        }
    }

    # Datos mock — lo que verá el científico
    mock_data = {
        "path": "/mock/data/path",
        "images": np.zeros((5, 256, 256, 3), dtype=np.uint8),
        "labels": [0, 1, 0, 1, 0],
        "train_config": {
            "batch_size": 32, "epochs": 500, "lr": 0.001,
            "patience": 50, "num_classes": 2
        }
    }

    image_asset = sy.Asset(name="classification_images", data=raw_data_path, mock=mock_data)
    medical_dataset = sy.Dataset(
        name="Medical Images Dataset",
        description="Total images: 1500. Labels available: ['class_A', 'class_B']. Preprocessing required."
    )
    medical_dataset.add_asset(image_asset)
    owner_client.upload_dataset(medical_dataset)
    print("Dataset subido correctamente.")


    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 2 — DATA SCIENTIST: envío de la solicitud de ejecución
    # ══════════════════════════════════════════════════════════════════════════════

    data_site = sy.orchestra.launch(name="Glaucoma-research-centre")
    scientist_client = data_site.login(email="rachel@datascience.inst", password="syftrocks")

    dataset = scientist_client.datasets["Medical Images Dataset"]
    asset   = dataset.assets["classification_images"]

    my_project = sy.Project(
        name="My TFG Project",
        description="Training a ResNet50 model remotely. Only metrics are returned.",
        members=[scientist_client]
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

        Solo se devuelven métricas; datos y modelo .pth quedan en el servidor.
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
                train, evaluate, SEED,
            )

            data_path   = Path(raw_data_dict["path"])
            cfg         = raw_data_dict.get("train_config", {})
            batch_size  = cfg.get("batch_size",  32)
            epochs      = cfg.get("epochs",      500)
            lr          = cfg.get("lr",           0.001)
            patience    = cfg.get("patience",     50)
            num_classes = cfg.get("num_classes",  2)

            set_seed(SEED)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 1. Preprocesado
            base_dataset = Data_Preprocessing(data_path=data_path, prep_batch_size=batch_size).dataset

            # 2. Split
            train_idx, val_idx, test_idx = split_dataset(base_dataset)
            train_split = base_dataset.select(train_idx)
            val_split   = base_dataset.select(val_idx)
            test_split  = Data_Preprocessing(data_path=data_path, prep_batch_size=batch_size, split_name='test').dataset


            # 3. Transforms
            train_split.set_transform(HFTransform(TrainProcessor()))
            val_split.set_transform(HFTransform(ValProcessor()))
            test_split.set_transform(HFTransform(ValProcessor()))

            # 4. DataLoaders
            sampler = make_balanced_sampler(base_dataset, train_idx)
            train_loader = DataLoader(train_split, batch_size=batch_size, sampler=sampler,
                                      collate_fn=collate_fn, num_workers=0, pin_memory=True)
            val_loader   = DataLoader(val_split,   batch_size=batch_size, shuffle=False,
                                      collate_fn=collate_fn, num_workers=0)
            test_loader  = DataLoader(test_split,  batch_size=batch_size, shuffle=False,
                                      collate_fn=collate_fn, num_workers=0)

            # 5. Modelo
            model = build_resnet50(num_classes=num_classes, pretrained=True).to(device)

            # 6. Entrenamiento
            best_val_acc = train(model, train_loader, val_loader, device,
                                 epochs=epochs, lr=lr, patience=patience)

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
                "test_samples":     len(test_split),
                "best_val_bal_acc": round(float(best_val_acc), 4),
                "test_bal_acc":     round(float(bal_acc),      4),
                "test_precision":   round(float(precision),    4),
                "test_recall":      round(float(recall),       4),
                "confusion_matrix": cm,
                "model_weights": {k: v.cpu() for k, v in model.state_dict().items()},
                "error":            None,
            }

        except Exception:
            return {"status": "CRASH INTERNO", "error": traceback.format_exc()}


    my_project.create_code_request(obj=remote_preprocessing_and_training, client=scientist_client)
    print("Solicitud enviada. Esperando aprobación del owner...")


    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 3 — DATA OWNER: revisión y aprobación
    # ══════════════════════════════════════════════════════════════════════════════

    owner_client = data_site.login(email=OWNER_EMAIL, password=OWNER_PASSWD)
    pending_requests = owner_client.requests
    print("\nSolicitudes pendientes:")
    for req in pending_requests:
        print(f"  Request ID: {req.id} | Status: {req.status}")

    incoming_request = pending_requests[0]
    print("\nCódigo de la solicitud:\n", incoming_request.code)
    incoming_request.approve()
    print("Solicitud aprobada.")


    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 4 — DATA SCIENTIST: ejecución y descarga de métricas
    # ══════════════════════════════════════════════════════════════════════════════

    scientist_client  = data_site.login(email="rachel@datascience.inst", password="syftrocks")
    result_pointer    = scientist_client.code.remote_preprocessing_and_training(raw_data_dict=asset)
    print("Ejecutando en el servidor remoto...")

    final_result = result_pointer.get()
    print("\nResultados recibidos por el científico:")
    print(f"  Status              : {final_result.get('status')}")
    print(f"  Dispositivo         : {final_result.get('device')}")
    print(f"  Total imágenes      : {final_result.get('total_imagenes')}")
    print(f"  Train / Val / Test  : {final_result.get('train_samples')} / "
          f"{final_result.get('val_samples')} / {final_result.get('test_samples')}")
    print(f"  Mejor Val Bal-Acc   : {final_result.get('best_val_bal_acc')}")
    print(f"  Test Bal-Acc        : {final_result.get('test_bal_acc')}")
    print(f"  Test Precision      : {final_result.get('test_precision')}")
    print(f"  Test Recall         : {final_result.get('test_recall')}")
    print(f"  Confusion matrix    : {final_result.get('confusion_matrix')}")
    model_weights = final_result["model_weights"]
    torch.save(model_weights, "best_rimone_syft.pth")
    if final_result.get("error"):
        print(f"\n  ERROR:\n{final_result['error']}")

    data_site.land()


if __name__ == "__main__":
    main()