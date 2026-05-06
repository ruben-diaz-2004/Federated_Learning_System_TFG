"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================

backdoor_syft.py
Entrenamiento federado en PySyft con ataque de backdoor.

Modo de uso:
    python backdoor_syft.py --data_dir /ruta/rimone_A --model_out best.pth
"""

import argparse

import numpy as np
import syft as sy
import torch

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
