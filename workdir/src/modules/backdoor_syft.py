# -*- coding: utf-8 -*-
"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
======================

backdoor_syft.py
Entrenamiento federado en PySyft con ataque de backdoor inyectado por el
data scientist malintencionado.

Escenario:
  - El owner publica el dataset a través de PySyft.
  - El scientist envía una "remote function" que parece un entrenamiento normal,
    pero en su interior llama a poison_dataset() y entrena con un trigger
    elegido (square, cross, gaussian, etc.).
  - El owner aprueba la solicitud creyendo que es un entrenamiento legítimo.
  - El modelo envenenado queda guardado en el servidor del owner; el
    scientist solo recibe métricas (incluyendo ASR, que delata el ataque
    pero solo si el owner audita los resultados).

Las claves del dict de retorno usan los mismos nombres que las columnas de
la tabla PoisoningRun (percent_poison, trigger_position) para que
register_poisoning_run() pueda hacer dict-unpacking directo.

Modo de uso:
    python backdoor_syft.py \
        --data_dir /ruta/rimone_A \
        --model_out backdoor_syft.pth \
        --trigger_type square \
        --percent_poison 0.2
"""

import argparse

import numpy as np
import syft as sy


def run_backdoor_syft_pipeline(
    data_dir:         str,
    model_out:        str   = "backdoor_syft.pth",
    cfg:              dict  = None,
    trigger_type:     str   = "square",
    percent_poison:   float = 0.2,
    trigger_size:     int   = 8,
    trigger_position: str   = "top_left",
    source_class:     int   = 1,
    target_class:     int   = 0,
    owner_email:      str   = "alu0101552613@ull.edu.es",
    owner_passwd:     str   = "glaucoma_research_syft_admin",
    scientist_email:  str   = "rachel@datascience.inst",
    scientist_passwd: str   = "syftrocks",
) -> dict:
    """
    Ejecuta el pipeline PySyft con backdoor y devuelve el dict de resultados.

    Parámetros
    ----------
    data_dir         : ruta al directorio del dataset en el servidor del owner.
    model_out        : nombre del fichero .pth donde se guardará el modelo
                       envenenado en el servidor del owner.
    cfg              : hiperparámetros de entrenamiento. Defaults:
                         batch_size=32, epochs=500, lr=0.001,
                         patience=50, num_classes=2.
    trigger_type     : square | cross | checkerboard | gaussian | sinusoidal | border
    percent_poison   : fracción de muestras de source_class a envenenar (0-1).
    trigger_size     : lado del trigger en píxeles. NULL en sinusoidal.
    trigger_position : top_left | top_right | bottom_left | bottom_right.
                       NULL en sinusoidal (trigger global).
    source_class     : clase origen — la que recibe el trigger.
    target_class     : clase objetivo — a la que redirige el trigger.

    Return
    ------
    dict con las claves alineadas a la tabla PoisoningRun:
        status, device,
        total_imagenes, train_samples, val_samples, test_samples,
        best_epoch, best_val_bal_acc,
        confusion_matrix, model_path, error,
        # ── Campos directos de PoisoningRun ──
        trigger_type, trigger_size, trigger_position,
        percent_poison, n_poisoned,
        source_class, target_class,
        clean_bal_acc, clean_precision, clean_recall,
        attack_success_rate.
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

    # Configuración del backdoor que viaja embutida en el asset.
    # En un escenario real el atacante hardcodearía estos valores dentro
    # de la función remota; aquí los pasamos por parámetro para facilitar
    # los experimentos comparativos entre triggers.
    _backdoor_cfg = {
        "trigger_type":     trigger_type,
        "percent_poison":   percent_poison,
        "trigger_size":     trigger_size,
        "trigger_position": trigger_position,
        "source_class":     source_class,
        "target_class":     target_class,
    }

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

    raw_data_path = {
        "path":         data_dir,
        "model_out":    model_out,
        "train_config": _cfg,
        "backdoor_cfg": _backdoor_cfg,
    }

    # Mock — lo que ve el científico
    mock_data = {
        "images":       np.zeros((5, 256, 256, 3), dtype=np.uint8),
        "labels":       [0, 1, 0, 1, 0],
        "train_config": _cfg,
        "backdoor_cfg": _backdoor_cfg,
    }

    image_asset = sy.Asset(
        name="classification_images", data=raw_data_path, mock=mock_data,
    )
    medical_dataset = sy.Dataset(
        name="Medical Images Dataset",
        description=(
            "Glaucoma research dataset. Labels available: ['Glaucoma', 'Normal']. "
            "Preprocessing required."
        ),
    )
    medical_dataset.add_asset(image_asset)
    owner_client.upload_dataset(medical_dataset)

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 2 — DATA SCIENTIST: envío de la solicitud de ejecución
    # ══════════════════════════════════════════════════════════════════════════
    data_site = sy.orchestra.launch(name="Glaucoma-research-centre")
    scientist_client = data_site.login(
        email=scientist_email, password=scientist_passwd,
    )

    dataset = scientist_client.datasets["Medical Images Dataset"]
    asset   = dataset.assets["classification_images"]

    my_project = sy.Project(
        name="My TFG Project (Backdoor)",
        description=(
            "Training a ResNet50 model remotely with a backdoor injected "
            "in the training pipeline. Only metrics are returned."
        ),
        members=[scientist_client],
    )

    @sy.syft_function(input_policy=sy.ExactMatch(raw_data_dict=asset))
    def remote_backdoor_training(raw_data_dict):
        """
        Pipeline de entrenamiento con backdoor inyectado:
          1. Data_Preprocessing  → HuggingFace Dataset
          2. split_dataset       → train / val / test
          3. dataset_to_numpy    → arrays para envenenar
          4. poison_dataset      → inyecta el trigger en source_class y
                                   reasigna su etiqueta a target_class
          5. build_resnet50      → ResNet50 preentrenado en ImageNet
          6. train()             → early stopping sobre VAL LIMPIA
          7. evaluate_loader     → métricas en test limpio
          8. compute_attack_success_rate → ASR sobre test envenenado

        El owner solo ve métricas en el dict de retorno; el modelo
        envenenado queda guardado en su disco bajo `model_out`.
        """
        import traceback
        try:
            import torch
            from pathlib import Path
            from sklearn.metrics import confusion_matrix

            from data_preprocessing import Data_Preprocessing
            from train_resnet import (
                set_seed, build_resnet50, split_dataset,
                HFTransform, TrainProcessor, ValProcessor,
                train,
            )
            from backdoor_attack import (
                get_trigger, dataset_to_numpy, poison_dataset,
                make_numpy_loader, evaluate_loader,
                compute_attack_success_rate,
            )

            # ── Parámetros recibidos del asset ───────────────────────────
            data_path  = Path(raw_data_dict["path"])
            model_out  = raw_data_dict.get("model_out", "backdoor_syft.pth")
            cfg        = raw_data_dict.get("train_config", {})
            bd_cfg     = raw_data_dict.get("backdoor_cfg",  {})

            batch_size  = cfg.get("batch_size",  32)
            epochs      = cfg.get("epochs",      500)
            lr          = cfg.get("lr",           0.001)
            patience    = cfg.get("patience",     50)
            num_classes = cfg.get("num_classes",  2)
            seed        = cfg.get("seed",         42)
            train_ratio = cfg.get("train_ratio",  0.70)
            val_ratio   = cfg.get("val_ratio",    0.10)

            trigger_type     = bd_cfg.get("trigger_type",     "square")
            percent_poison   = bd_cfg.get("percent_poison",   0.2)
            trigger_size     = bd_cfg.get("trigger_size",     8)
            trigger_position = bd_cfg.get("trigger_position", "top_left")
            source_class     = bd_cfg.get("source_class",     1)
            target_class     = bd_cfg.get("target_class",     0)

            set_seed(seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ── 1. Dataset y split ───────────────────────────────────────
            base_dataset = Data_Preprocessing(
                data_path=data_path, prep_batch_size=batch_size,
            ).dataset
            train_idx, val_idx, test_idx = split_dataset(
                base_dataset, train_ratio, val_ratio,
            )
            train_split = base_dataset.select(train_idx)
            val_split   = base_dataset.select(val_idx)
            test_split  = base_dataset.select(test_idx)

            train_split.set_transform(HFTransform(TrainProcessor()))
            val_split.set_transform(HFTransform(ValProcessor()))
            test_split.set_transform(HFTransform(ValProcessor()))

            # ── 2. Arrays numpy para envenenamiento ──────────────────────
            x_train, y_train = dataset_to_numpy(train_split, batch_size)
            x_val,   y_val   = dataset_to_numpy(val_split,   batch_size)
            x_test,  y_test  = dataset_to_numpy(test_split,  batch_size)

            # ── 3. Envenenamiento ────────────────────────────────────────
            trigger_fn = get_trigger(
                trigger_type, size=trigger_size, position=trigger_position,
            )
            x_train_p, y_train_p, n_poisoned, _ = poison_dataset(
                x_train, y_train,
                trigger_fn=trigger_fn,
                poison_rate=percent_poison,   # nombre interno en backdoor_attack.py
                source_class=source_class,
                target_class=target_class,
                seed=seed,
            )

            # ── 4. DataLoaders ───────────────────────────────────────────
            train_loader = make_numpy_loader(
                x_train_p, y_train_p, batch_size, balanced=True,
            )
            val_loader   = make_numpy_loader(x_val,  y_val,  batch_size)
            test_loader  = make_numpy_loader(x_test, y_test, batch_size)

            # ── 5. Modelo y entrenamiento ────────────────────────────────
            model = build_resnet50(
                num_classes=num_classes, pretrained=True,
            ).to(device)
            best_val_bal_acc, best_epoch = train(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=lr, patience=patience,
                save_path=model_out,
            )

            model.load_state_dict(
                torch.load(model_out, map_location=device, weights_only=True)
            )

            # ── 6. Evaluación sobre test limpio ──────────────────────────
            clean_bal, clean_prec, clean_rec = evaluate_loader(
                model, test_loader, device, label="limpio",
            )

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
            cm = confusion_matrix(all_labels, all_preds).tolist()

            # ── 7. Attack Success Rate ───────────────────────────────────
            asr = compute_attack_success_rate(
                model, x_test, y_test,
                trigger_fn=trigger_fn,
                source_class=source_class,
                target_class=target_class,
                batch_size=batch_size,
                device=device,
            )

            # Para sinusoidal, el schema permite NULL en trigger_size y
            # trigger_position. Lo reflejamos aquí.
            ts_out = None if trigger_type == "sinusoidal" else int(trigger_size)
            tp_out = None if trigger_type == "sinusoidal" else trigger_position

            return {
                "status":              "Exito",
                "device":              str(device),
                "total_imagenes":      len(base_dataset),
                "train_samples":       len(train_idx),
                "val_samples":         len(val_idx),
                "test_samples":        len(test_idx),
                "best_epoch":          int(best_epoch),
                "best_val_bal_acc":    round(float(best_val_bal_acc), 4),
                "confusion_matrix":    cm,
                "model_path":          model_out,
                "error":               None,
                # ── Campos alineados con PoisoningRun ──
                "trigger_type":        trigger_type,
                "trigger_size":        ts_out,
                "trigger_position":    tp_out,
                "percent_poison":      float(percent_poison),
                "n_poisoned":          int(n_poisoned),
                "source_class":        int(source_class),
                "target_class":        int(target_class),
                "clean_bal_acc":       round(float(clean_bal),  4),
                "clean_precision":     round(float(clean_prec), 4),
                "clean_recall":        round(float(clean_rec),  4),
                "attack_success_rate": round(float(asr),        4),
            }

        except Exception:
            return {"status": "CRASH INTERNO", "error": traceback.format_exc()}

    my_project.create_code_request(
        obj=remote_backdoor_training, client=scientist_client,
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
    scientist_client = data_site.login(
        email=scientist_email, password=scientist_passwd,
    )

    result_pointer = scientist_client.code.remote_backdoor_training(
        raw_data_dict=asset,
    )
    print("Ejecutando en el servidor remoto...")

    final_result = result_pointer.get()

    print("\nResultados recibidos por el científico:")
    print(f"  Status              : {final_result.get('status')}")
    print(f"  Dispositivo         : {final_result.get('device')}")
    print(f"  Total imágenes      : {final_result.get('total_imagenes')}")
    print(f"  Train / Val / Test  : {final_result.get('train_samples')} / "
          f"{final_result.get('val_samples')} / {final_result.get('test_samples')}")
    print(f"  Trigger             : {final_result.get('trigger_type')} "
          f"(size={final_result.get('trigger_size')}, "
          f"pos={final_result.get('trigger_position')})")
    print(f"  % poison            : {final_result.get('percent_poison')}")
    print(f"  Muestras envenenadas: {final_result.get('n_poisoned')}")
    print(f"  Source → target     : {final_result.get('source_class')} → "
          f"{final_result.get('target_class')}")
    print(f"  Mejor época         : {final_result.get('best_epoch')}")
    print(f"  Mejor Val Bal-Acc   : {final_result.get('best_val_bal_acc')}")
    print(f"  Test Bal-Acc limpio : {final_result.get('clean_bal_acc')}")
    print(f"  Test Precision      : {final_result.get('clean_precision')}")
    print(f"  Test Recall         : {final_result.get('clean_recall')}")
    print(f"  Attack Success Rate : {final_result.get('attack_success_rate')}")
    print(f"  Confusion matrix    : {final_result.get('confusion_matrix')}")
    print(f"  Modelo guardado en  : {final_result.get('model_path')}")

    if final_result.get("error"):
        print(f"\n  ERROR:\n{final_result['error']}")

    data_site.land()
    return final_result


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline PySyft de entrenamiento federado con backdoor.",
    )
    parser.add_argument("--data_dir",     required=True,
                        help="Ruta al directorio del dataset")
    parser.add_argument("--model_out",    default="backdoor_syft.pth",
                        help="Fichero .pth de salida del modelo envenenado")
    # Hiperparámetros de entrenamiento
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--patience",     type=int,   default=50)
    parser.add_argument("--num_classes",  type=int,   default=2)
    # Backdoor
    parser.add_argument("--trigger_type",     type=str, default="square",
                        choices=["square", "cross", "checkerboard",
                                 "gaussian", "sinusoidal", "border"])
    parser.add_argument("--percent_poison",   type=float, default=0.2)
    parser.add_argument("--trigger_size",     type=int,   default=8)
    parser.add_argument("--trigger_position", type=str,   default="top_left",
                        choices=["top_left", "top_right",
                                 "bottom_left", "bottom_right"])
    parser.add_argument("--source_class",     type=int,   default=1)
    parser.add_argument("--target_class",     type=int,   default=0)
    # Credenciales
    parser.add_argument("--owner_email",      default="alu0101552613@ull.edu.es")
    parser.add_argument("--owner_passwd",     default="glaucoma_research_syft_admin")
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

    run_backdoor_syft_pipeline(
        data_dir         = args.data_dir,
        model_out        = args.model_out,
        cfg              = cfg,
        trigger_type     = args.trigger_type,
        percent_poison   = args.percent_poison,
        trigger_size     = args.trigger_size,
        trigger_position = args.trigger_position,
        source_class     = args.source_class,
        target_class     = args.target_class,
        owner_email      = args.owner_email,
        owner_passwd     = args.owner_passwd,
        scientist_email  = args.scientist_email,
        scientist_passwd = args.scientist_passwd,
    )


if __name__ == "__main__":
    main()
