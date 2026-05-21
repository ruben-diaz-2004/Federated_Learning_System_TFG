from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from time import sleep
from threading import current_thread, Thread, Event
import io
import syft as sy
from syft.service.policy.policy import MixedInputPolicy
import torch

from create_dataset import create_dataset
from database_access import (
    register_server, register_attack_experiment, register_experiment_split,
    register_split_server, save_federated_round_results,
)
from database_access import get_experiment, get_experiment_splits, get_server, get_dataset

######################################################
# Configurando un experimento con un dict
######################################################

EVE_EMAIL="eve@codigla.org"
EVE_NAME="Eve"
EVE_PASSWORD="changethis"
EVE_INSTITUTION="Eve Place"
EVE_WEBSITE="www.eveplace.org"

EXP_CONFIG = {
        "name": "training_refuge_rimone_v1",
        "create_dataset" : True,

        "datasets" : [ \
                { 
                    "path" : "/mnt/nvme/tfg0_workspace/tfg0/Federated_Learning_System_TFG/refuge_x",
                    "name" : "refuge",
                    "zip_paths" : ["/mnt/nvme/tfg0_workspace/tfg0/Federated_Learning_System_TFG/datos_base/refuge_real_glaucoma.zip","/mnt/nvme/tfg0_workspace/tfg0/Federated_Learning_System_TFG/datos_base/refuge_real_normal.zip"],
                    "class_names" : ["glaucoma","normal"]
                },
                { 
                    "path" : "/mnt/nvme/tfg0_workspace/tfg0/Federated_Learning_System_TFG/rimone_x",
                    "name" : "rimone",
                    "zip_paths" : ["/mnt/nvme/tfg0_workspace/tfg0/Federated_Learning_System_TFG/datos_base/nrimone_real_glaucoma.zip","/mnt/nvme/tfg0_workspace/tfg0/Federated_Learning_System_TFG/datos_base/nrimone_real_normal.zip"],
                    "class_names" : ["glaucoma","normal"]
                }],

        "servers" : [ \
                {
                    "name" : "server_alice",
                    "owner_email" : "alice@codigla.org",
                    "owner_password" : "changethis",
                    "owner_name" : "Alice",
                    "owner_institution" : "Alice Place"
                },
                {
                    "name" : "server_bob",
                    "owner_email" : "bob@codigla.org",
                    "owner_password" : "changethis",
                    "owner_name" : "Bob",
                    "owner_institution" : "Bob Place"
                }
                ],

        "splits": [ \
                {
                    "dataset_name" : "refuge",
                    "server_name" : "server_alice",
                    "model_path" : "./model_refuge/split1/model.pth",
                    "train_ratio" : 0.7,
                    "val_ratio" : 0.1,
                    "seed" : 0
                },
                {
                    "dataset_name" : "rimone",
                    "server_name" : "server_bob",
                    "model_path" : "./model_rimone/split1/model.pth",
                    "train_ratio" : 0.7,
                    "val_ratio" : 0.1,
                    "seed" : 0
                }
                ],

        "common_training": {
            "batch_size" : 32,
            "epochs" : 3,
            "lr" : 1e-4,
            "patience" : 50,
            "eve_model_path" : "./eve_model.pth"
            },

        "description" : "Simple training in federated learning"

        
}

#############################################################
# Funciones auxiliares
# Ver tutorial: github.com/OpenMined/syft-heart-disease-tutorial
##############################################################

# Launch servers : from syft heart disease tutorial
# Crear un syft dataseta: variaci'on para trabajar solo con paths y mock data

def create_syft_dataset(dataset_info):
    data_path=dataset_info["path"]
    data_description = dataset_info["description"]

    # Mock generation
    mock_data_path=dataset_info["mock_path"]
    zip_file_bytes_io = io.BytesIO()
    empty_file=io.BytesIO()
    empty_file.write(b"")
    #TODO GENERATE A ZIP FILE WITH TWO DIRECTORIES AND A NUMBER OF RANDOM PNG IMAGES
    with ZipFile(zip_file_bytes_io, 'w') as zip_file:
        zip_file.writestr(str(Path('./') / mock_data_path / "dummy.png"), empty_file.getvalue())

    mock_content=zip_file_bytes_io

    mock_description="""

    Run the following code to produce the data directory.
    Data directory path is the input to the processing function.


    archive = ZipFile(BytesIO(mock["content"]))
    archive.extractall(root)

    """
    raw_data = {
            "path": data_path
            }

    mock_data = {
            "path" : mock_data_path,
            "content" : mock_content,
            }

    asset = sy.Asset(name="directory_binary_classification",
            data=raw_data,
            mock=mock_data)

    dataset= sy.Dataset(name=dataset_info["name"],
            description=dataset_info["description"],
            asset_list=[asset]
            )

    return dataset

# Clase para un thread que se pueda detener y que se pueda consultar si est'a parado
# Para aprovar automaticamente peticiones

class DataSiteThread(Thread):
    """
    Thread class with a stop() method.
    The thread itself has to check regularly for the stopped() condition.

    See here:
    https://stackoverflow.com/questions/47912701/python-how-can-i-implement-a-stoppable-thread
    """

    def __init__(self, *args, **kwargs):
        super(DataSiteThread, self).__init__(*args, **kwargs)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


# Funcion para ser lanzada como thread y aprobar automaticamente peticiones
# client debe ser un owner client
def check_and_approve_incoming_requests(client):
    """This utility function will set the server in busy-waiting
    to constantly check and auto-approve any incoming code requests.

    Note: This function is only intended for the tutorial as demonstration
    of the PoC example.
    For further information about please check out the official for the
    Requests API: https://docs.openmined.org/en/latest/components/requests-api.html
    """
    while not current_thread().stopped():  # type: ignore
        requests = client.requests
        for r in filter(lambda r: r.status.value != 2, requests):  # 2 == APPROVED
            r.approve(approve_nested=True)
            # print("New Request approved in ")
        sleep(1)

# Funci'on para lanzar un servidor local
def spawn_server(server_info,dataset_info):
    """Utility function to launch a new instance of a PySyft Datasite"""

    data_site = sy.orchestra.launch(
        name=server_info["name"],
        port=server_info["port"],
        reset=True,
        n_consumers=3,
        create_producer=True,
    )
    owner_client = data_site.login(email="info@openmined.org", password="changethis")
    owner_client.account.set_email(server_info["owner_client_email"])
    owner_client.account.set_password(server_info["owner_client_password"],confirm=False)
    owner_client.account.update(
            name=server_info["owner_client_name"],
            institution=server_info["owner_client_institution"]
            )



    

    # Customise Settings
    #client.settings.allow_guest_signup(True)
    #client.settings.welcome_customize(
    #    markdown=_get_welcome_message(name=name, full_name=INSTITUTE_FULLNAMES[name])
    #)

    owner_client.users.create(
        email=EVE_EMAIL,
        password=EVE_PASSWORD,
        password_verify=EVE_PASSWORD,
        name=EVE_NAME,
        institution=EVE_INSTITUTION,
        website=EVE_WEBSITE,
    )

    user = owner_client.users[-1]
    # user.allow_mock_execution(True)

    ds = create_syft_dataset(dataset_info)
    if not ds is None:
        owner_client.upload_dataset(ds)

    print(f"Datasite {server_info['name']} is up and running: {data_site.url}:{data_site.port}")
    return data_site, owner_client


# Funcion que puede ser lanzada en un Thread para tener un pull de servidores
def launch_datasites(server_info_list=[], dataset_info_list=[],event=Event(),show : bool = True) -> None:
    # Can be launched from thread

    if not server_info_list:
        raise Exception("Trying to launch datasites without server info")

    data_sites = list()
    client_threads = list()
    for i,server_info in enumerate(server_info_list):
        data_site, client = spawn_server(server_info,dataset_info_list[i])
        data_sites.append(data_site)
        client_threads.append(
            DataSiteThread(
                target=check_and_approve_incoming_requests, args=(client,), daemon=True
            )
        )
    for t in client_threads:
        t.start()

    def finish():
        # Signal threads to stop
        for client_thread in client_threads:
            client_thread.stop()
            
        # Wait for threads to actually finish their current loop
        for client_thread in client_threads:
            client_thread.join()

        # Safely land the data sites
        for data_site in data_sites:
            data_site.land()

    #if show_conn_info:
    #    show_connections_info()
    stop_flag=False
    try:

        while not stop_flag:
            sleep(2)
            if event.is_set():
                stop_flag=True

        finish()

    except KeyboardInterrupt:
        finish()


def remote_preprocessing_and_training(raw_data_dict,model_params):
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
            train_epochs, evaluate, recalculate_norm_weights
        )

        data_path   = Path(raw_data_dict["path"])
        state       = model_params.get("state")
        eve_model_path    = model_params.get("eve_model_path")
        model_out   = model_params.get("model_out", "best_rimone_syft.pth")
        epochs      = model_params.get("epochs",1)
        cfg         = model_params.get("train_config", {})
        batch_size  = cfg.get("batch_size",  32)
        lr          = cfg.get("lr",           0.001)
        patience    = cfg.get("patience",     50)
        num_classes = cfg.get("num_classes",  2)
        seed        = cfg.get("seed",         42)
        train_ratio = cfg.get("train_ratio",  0.70)
        val_ratio   = cfg.get("val_ratio",    0.10)

        #TODO: initialize eve_model_path

        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        # 1. Preprocesado
        base_dataset = Data_Preprocessing(
            data_path=data_path, prep_batch_size=batch_size,num_proc=8
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
        if state==0: # Estado inicial, partimos de los pesos del modelo preentrenado
           model = build_resnet50(num_classes=num_classes, pretrained=True).to(device)
        else:
           model = build_resnet50(num_classes=num_classes, pretrained=False).to(device)
           model.load_state_dict(torch.load(eve_model_path,weights_only=True))

        model=model.to(device)

        val_bal_acc=None
        loss=None
        bal_acc=None
        precision=None
        recall=None
        cm=None


        if state!=2: # Entrenamiento
            if state!=0:
                # Recalculamos pesos de normalizaci'on y similares
                recalculate_norm_weights(model,train_loader,device)
            # 6. Entrenamiento — train() devuelve (best_val_acc, best_epoch)
            val_bal_acc, loss = train_epochs(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=lr, 
                save_path=model_out,
            )
        else:
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
            "val_bal_acc":      val_bal_acc,
            "loss" :            loss,
            "test_bal_acc":     bal_acc,
            "test_precision":   precision,
            "test_recall":      recall, 
            "confusion_matrix": cm,
            "model_path":       model_out,
            "error":            None,
        }
 
    except Exception:
         return {"status": "CRASH INTERNO", "error": traceback.format_exc()}
 
def fl_compute_model(in_models=[],out_model="./model.pth"):

     if not in_models:
        raise Exception("Input models list is empty") 

     # M'etodo average simple, el 'unico implementado
     first=True
     # Calculo de los pesos
     w_per_model = [float(result["train_samples"]) for (_,result) in in_models]
     s=sum(w_per_model)
     w_per_model = [ x/s for x in w_per_model ]
     print(f"Weights: {w_per_model}")
     for (i,(model_path, result)) in enumerate(in_models):
         # Load
         state_dict=torch.load(model_path)
         for key in state_dict.keys():
             state_dict[key]=state_dict[key]*w_per_model[i]
         if first:
             model=state_dict
         else:
             # Promedio de los pesos
             for key in state_dict.keys():
                 if not key in model.keys():
                     raise Exception("Averaging model weights with different keys is not possible")
                 model[key]+=state_dict[key]
         first=False
     #Save    
     torch.save(model,out_model)
        
 


def main():

    # Creaci'on del experimento
    cfg = EXP_CONFIG["common_training"]
    experiment_id = register_attack_experiment(
            name            = EXP_CONFIG["name"],
            experiment_type = "training",
            eve_model_path  = cfg["eve_model_path"],
            lr              = cfg["lr"],
            batch_size      = cfg["batch_size"],
            epochs_max      = cfg["epochs"],
            patience        = cfg["patience"],
            description     = EXP_CONFIG["description"],
            )

    # Creaci'on de datasets
    if EXP_CONFIG["create_dataset"]:
        for dataset in EXP_CONFIG["datasets"]:
            print(dataset)
            create_dataset(
                    dataset_dir=Path(dataset["path"]),
                    dataset_name=dataset["name"],
                    zip_paths=[Path(z) for z in dataset["zip_paths"]],
                    class_names=dataset["class_names"])


    # Creaci'on de servidores de datos
    for server in EXP_CONFIG["servers"]:
        register_server(
                server["name"],
                server["owner_email"],
                server["owner_password"])

    # Creaci'on de splits para el experimento

    for split in EXP_CONFIG["splits"]:

        split_id=register_split_server(split["dataset_name"],
                split["server_name"],
                split["model_path"],
                n_train=-1, n_val=-1, n_test=-1, # Still pending
                seed=split["seed"],
                train_ratio=split["train_ratio"],
                val_ratio=split["val_ratio"])

        if split_id is not None:
            register_experiment_split(experiment_id,split_id)
        else:
            raise(f"Error creating split for dataset {split['dataset_name']} and server {split['server_name']}")

    # Ahora lanzamos el experimento

    # Fase 1: Configuracion

    # Configuraci'on del entrenamiento
    
    cfg_experiment=get_experiment(experiment_id)

    # 1. Obtenemos servidores involucrados
    splits_by_server=get_experiment_splits(experiment_id)

    # Main loop: mientras tengamos alg'un servidor con splits pendientes

    splits_pending=True

    while(splits_pending):

        server_info_list=[]
        dataset_info_list=[]
        split_info_list=[]
        event=Event()
        port=cfg_experiment["servers_base_port"]


        for server_id in [int(x) for x in splits_by_server.keys()]:

            if not splits_by_server[str(server_id)]:
                continue

            # Split a procesar en este aprendizaje federado
            split_info=dict(splits_by_server[str(server_id)][0])


            #Split consumed
            splits_by_server[str(server_id)]=splits_by_server[str(server_id)][1:]
            
            server_info=get_server(server_id)
            server_info["port"]=port

            port +=1
            
            dataset_info=get_dataset(split_info["dataset_id"])

            server_info_list.append(server_info)
            dataset_info_list.append(dataset_info)
            split_info_list.append(split_info)

        if server_info_list:
            print("Launching servers....")


            t = Thread(target=launch_datasites, 
                    kwargs={"server_info_list" : server_info_list,
                            "dataset_info_list" : dataset_info_list,
                            "event" : event,
                            "show": False}, daemon=True)
            t.start()
            # Fase de entrenamiento
            print("Wait for servers start")
            sleep(5)

            # Crear el UserCode y solicitar su ejecucion

            for (pos,server_info) in enumerate(server_info_list):
                print(f"Trying to log  in server {server_info['name']}")
                eve_client = sy.login(url="localhost",
                            port=server_info["port"],
                            email=EVE_EMAIL,
                            password=EVE_PASSWORD)
                data_asset=eve_client.datasets[0].assets[0]
                # Create UserCode
                syft_experiment = sy.syft_function(
                        input_policy=MixedInputPolicy(
                        client=eve_client, 
                        raw_data_dict=data_asset, 
                        model_params=dict))(remote_preprocessing_and_training)

                # Send request for UserCode
                eve_client.code.request_code_execution(syft_experiment)




            # Fase de entrenamiento
            sleep(5) # Espera por aprobacion user codes

            state=0 # Comenzamos en la primera epoca inicializando los pesos del modelo preentrenado

            for epoch in range(cfg_experiment["epochs_max"]):
                model_list=[]
                # For each server
                for (pos,server_info) in enumerate(server_info_list):
                    split_info=split_info_list[pos]
                    dataset_info=dataset_info_list[pos]
                    train_config={
                            "batch_size" : cfg_experiment["batch_size"],
                            "lr" : cfg_experiment["lr"],
                            "num_classes" : 2,
                            "seed" :  split_info["seed"],
                            "train_ratio" : split_info["train_ratio"],
                            "val_ratio" : split_info["val_ratio"]
                            }


                    model_params={}
                    model_params["state"]=state 
                    model_params["eve_model_path"]=cfg_experiment["eve_model_path"]
                    model_params["model_out"]=split_info["model_path"]
                    model_params["epochs"]=3
                    
                    model_params["train_config"]=train_config

                    print(f"Trying to log  in server {server_info['name']}")
                    eve_client = sy.login(url="localhost",
                            port=server_info["port"],
                            email=EVE_EMAIL,
                            password=EVE_PASSWORD)
                    data_asset=eve_client.datasets[0].assets[0]

                    # Create UserCode
                    #syft_experiment = sy.syft_function(
                    #    input_policy=MixedInputPolicy(
                    #    client=eve_client, 
                    #    raw_data_dict=data_asset, 
                    #    model_params=dict))(remote_preprocessing_and_training)

                    # Send request for UserCode
                    #eve_client.code.request_code_execution(syft_experiment)

                    #print("Waiting for request approval")
                    #sleep(5)
                    #eve_client.code.get_all()

                    # Execute
                    result=eve_client.code.remote_preprocessing_and_training(
                            raw_data_dict=data_asset,
                            model_params=model_params).get_from(eve_client)
                    print(result)
                    # Un nuevo modelo calculado
                    model_list.append((model_params["model_out"],dict(result)))

                fl_compute_model(in_models=model_list,out_model=cfg_experiment["eve_model_path"])
                state=1 # Now we use the averaged eve_model in subsequent epochs

            # -- Persist federated round results ----------------------------
            # One TrainingResult row is shared by all splits in this round.
            # Metrics are weighted averages across servers (by train_samples).
            result_id = save_federated_round_results(
                model_list            = model_list,
                split_info_list       = split_info_list,
                epochs_completed      = cfg_experiment["epochs_max"],
                aggregated_model_path = cfg_experiment["eve_model_path"],
            )
            print(f"[DB] Round results saved: result_id={result_id} "
                  f"linked to {len(split_info_list)} split(s)")

            # ── Shutdown servers ───────────────────────────────────────────
            print("Signaling servers to shut down...")
            sleep(5)
            event.set()

            # Wait for the background thread to cleanly exit
            t.join()
            print("All servers safely shut down. Exiting program.")






        else:
            splits_pending=False


"""
        ## Fase de entrenamiento

        for (data_site,split_id) in datasites_for_fed_training:

            # Petici'on para cada data site

            scientist_client = data_site.login(
                    email=EVE_EMAIL, password=EVE_PASSWORD
                    )

            dataset = scientis_client.datasets["Medical Images Dataset"]
            asset = dataset.assets["classification_images"]
            mock_data=asset.mock

            eve_model_path=mock_data["train_config"]["eve_model_path"]
            train_nsamples=mock_data["train_samples"]

            my_project = sy.Project(
                    name="My TFG Project",
                    description="Training a ResNet50 model remotely. Only metrics are returned.",
                    members=[scientist_client],
                )

            my_project.create_code_request(
                    obj=remote_preprocessing_and_training, client=scientist_client
                )

            # Aprobaci'on de la peticion

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









            










        




    # 4. Para cada servidor reamos mock_data

    # 5. Para cada servidor hacemos la carga de los datos.

    # Fase 2: Algoritmo de Federated Learning por promedios

    # 6. Para cada servidor Eve hace la petición de carga de pesos y orden entrenar o testear. 

    #  Si la orden es entrenar.


    # 7. Para cada servidor Eve pide entrenar durante n epocas. Eve recibe pesos y datos de validacion.

    # 8. Eve recoge los pesos y construyo un conjunto de pesos nuevo.

    # 9. Eve toma la decisi'on de temrinar el entrenamiento o proseguir volviendo al paso 6, con la orden de evaluar o testear.

    # Si la orden es evaluar

    # 7. Para cada servidor Eve pide evaluar el actual conjunto de pesos en el conjunto de test.



    

"""

        

if __name__ == "__main__":
    main()