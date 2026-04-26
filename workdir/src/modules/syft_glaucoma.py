# -*- coding: utf-8 -*-
"""
syft_glaucoma.py
================
Pipeline completo PySyft para el proyecto de investigación de glaucoma.
Combina los flujos del Data Owner y del Data Scientist (cliente).

Orden de ejecución:
  1. OWNER  (setup)    — Lanzar servidor, configurar cuenta, subir dataset
  2. SCIENTIST         — Conectarse, explorar dataset, definir y enviar función remota
  3. OWNER  (revisión) — Revisar la solicitud entrante y aprobarla
  4. SCIENTIST         — Ejecutar la función aprobada y descargar resultados
"""

import numpy as np
import syft as sy

print(sy.__version__)

def main():
    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 1 — DATA OWNER: setup del servidor y subida del dataset
    # ══════════════════════════════════════════════════════════════════════════════

    # ── 1.1 Lanzar el servidor de datos ──────────────────────────────────────────
    data_site = sy.orchestra.launch(name="Glaucoma-research-centre", reset=True)

    # ── 1.2 Login como root con credenciales por defecto ─────────────────────────
    owner_client = data_site.login(email="info@openmined.org", password="changethis")

    # ── 1.3 Actualizar credenciales del owner ────────────────────────────────────
    OWNER_EMAIL  = "alu0101552613@ull.edu.es"
    OWNER_PASSWD = "glaucoma_research_syft_admin"

    owner_client.account.set_email(OWNER_EMAIL)
    owner_client.account.set_password(OWNER_PASSWD, confirm=False)

    owner_client.account.update(
        name="Ruben, the Data Owner",
        institution="Glaucoma Research Centre"
    )

    # ── 1.4 Crear cuenta del científico de datos ─────────────────────────────────
    rachel_account_info = owner_client.users.create(
        email="rachel@datascience.inst",
        name="Dr. Rachel Science",
        password="syftrocks",
        password_verify="syftrocks",
        institution="Data Science Institute",
        website="https://datascience_institute.research.data"
    )

    print(owner_client.users)

    # ── 1.5 Definir y subir el dataset ───────────────────────────────────────────

    # Puntero a los datos reales en el servidor (solo visible para el owner)
    raw_data_path = {
        "path": "C:\\Users\\ruben\\Desktop\\TFG\\workdir\\trial_0"
    }

    # Datos mock (públicos) que el científico verá en lugar de los datos reales
    mock_data = {
        "images": np.zeros((5, 256, 256, 3), dtype=np.uint8),  # 5 imágenes negras 256×256 RGB
        "labels": [0, 1, 0, 1, 0]
    }

    image_asset = sy.Asset(
        name="classification_images",
        data=raw_data_path,
        mock=mock_data
    )

    medical_dataset = sy.Dataset(
        name="Medical Images Dataset",
        description=(
            "Total images: 1500. "
            "Labels available: ['class_A', 'class_B']. "
            "Preprocessing required."
        )
    )
    medical_dataset.add_asset(image_asset)

    owner_client.upload_dataset(medical_dataset)
    print("Dataset y Asset subidos correctamente al servidor PySyft.")

    print(owner_client.datasets)

    # Verificar el contenido del asset (solo visible para el owner)
    test_asset = medical_dataset.assets[0]
    print("data  →", test_asset.data)
    print("mock  →", test_asset.mock)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 2 — DATA SCIENTIST: envío de la solicitud de ejecución
    # ══════════════════════════════════════════════════════════════════════════════

    # ── 2.1 Conectarse al servidor ───────────────────────────────────────────────
    data_site = sy.orchestra.launch(name="Glaucoma-research-centre")
    scientist_client = data_site.login(email="rachel@datascience.inst", password="syftrocks")

    # ── 2.2 Explorar el dataset y obtener el puntero al asset ────────────────────
    dataset = scientist_client.datasets["Medical Images Dataset"]
    asset   = dataset.assets["classification_images"]

    data_pointer = asset.pointer
    print("Puntero a los datos en el servidor:", data_pointer)

    # ── 2.3 Crear el proyecto PySyft ─────────────────────────────────────────────
    my_project = sy.Project(
        name="My TFG Project",
        description="Training a classification model using remote data preprocessing.",
        members=[scientist_client]
    )

    # ── 2.4 Definir la función remota y enviar la solicitud al owner ──────────────
    @sy.syft_function(input_policy=sy.ExactMatch(raw_data_dict=asset))
    def remote_preprocessing_and_training(raw_data_dict):
        """
        Función que se ejecuta en el servidor del owner.
        Recibe el diccionario con la ruta real, instancia Data_Preprocessing
        y devuelve estadísticas básicas del dataset procesado.
        """
        import traceback

        try:
            from workdir.src.modules.data_preprocessing import Data_Preprocessing

            actual_path  = raw_data_dict["path"]
            preprocessor = Data_Preprocessing(
                data_path=actual_path,
                prep_batch_size=32
            )

            ready_dataset = preprocessor.dataset
            num_imagenes  = len(ready_dataset)

            return {
                "status": "Exito",
                "total_imagenes": num_imagenes,
                "error": None
            }

        except Exception as e:
            return {
                "status": "CRASH INTERNO",
                "error": traceback.format_exc()
            }


    my_project.create_code_request(obj=remote_preprocessing_and_training, client=scientist_client)
    print("Solicitud de ejecución enviada. Esperando aprobación del owner...")


    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 3 — DATA OWNER: revisión y aprobación de la solicitud
    # ══════════════════════════════════════════════════════════════════════════════

    # Re-login del owner (por si la sesión ha caducado)
    owner_client = data_site.login(email=OWNER_EMAIL, password=OWNER_PASSWD)

    pending_requests = owner_client.requests
    print("\nSolicitudes pendientes:")
    for req in pending_requests:
        print(f"  Request ID: {req.id} | Status: {req.status}")

    # Tomar la solicitud más reciente
    incoming_request = pending_requests[0]

    # Inspeccionar el código antes de aprobar
    print("\nCódigo de la solicitud:\n")
    print(incoming_request.code)

    # Aprobar la solicitud
    incoming_request.approve()
    print("Solicitud aprobada. El científico puede ejecutar su código.")


    # ══════════════════════════════════════════════════════════════════════════════
    # SECCIÓN 4 — DATA SCIENTIST: ejecución remota y descarga de resultados
    # ══════════════════════════════════════════════════════════════════════════════

    # Re-login del científico
    scientist_client = data_site.login(email="rachel@datascience.inst", password="syftrocks")

    approved_function = scientist_client.code.remote_preprocessing_and_training
    result_pointer    = approved_function(raw_data_dict=asset)
    print("Código ejecutándose en el servidor remoto...")

    # Descargar el resultado (solo los datos permitidos viajan al científico)
    final_result = result_pointer.get()
    print("\nEjecución finalizada. Resultados:")
    print(final_result)

    # ── Apagar el servidor ───────────────────────────────────────────────────────
    data_site.land()

if __name__ == "__main__":
    main()