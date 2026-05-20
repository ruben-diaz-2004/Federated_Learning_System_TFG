"""
@author: Rubén Díaz Marrero
Grado en ingeniería informática, Universidad de La Laguna
Trabajo de Fin de Grado — Curso 2025/2026
    ======================
    Módulo de acceso a la base de datos glaucoma_ml.

    La relación experimento ↔ split se gestiona mediante
    ExperimentSplit (tabla pivote), lo que permite evaluar
    un mismo experimento sobre varios splits distintos.

    Flujo típico:
      1. register_dataset(...)         → dataset_id
      2. register_split(...)           → split_id   (una o varias veces)
      3. register_experiment(...)      → experiment_id
      4. register_experiment_split(experiment_id, split_id)  → es_id
      5. register_training_result(...) → result_id
      6. link_result_to_es(es_id, result_id)
      7. register_adversarial_run(...) → adv_id
"""

import mysql.connector
from contextlib import contextmanager

DB_CONFIG = {
    "host":     "localhost",
    "user":     "tfg0",
    "password": "password",
    "database": "federate",
    "port":     2200,
}

@contextmanager
def get_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────
def register_dataset(name: str, path: str,
                     total_samples: int,
                     samples_class0: int,
                     samples_class1: int) -> int:
    """
    Inserta un dataset y devuelve su dataset_id.
    Si el nombre ya existe devuelve el id existente (idempotente).

    Ejemplo:
        dataset_id = register_dataset(
            name="REFUGE",
            path="/data/refuge_x",
            total_samples=1200,
            samples_class0=1080,
            samples_class1=120,
        )
    """
    sql = """
        INSERT INTO Dataset (name, path, total_samples, samples_class0, samples_class1)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE dataset_id = LAST_INSERT_ID(dataset_id)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (name, path, total_samples, samples_class0, samples_class1))
        return cur.lastrowid


# ─────────────────────────────────────────────────────────────
# 2. Split 
# ─────────────────────────────────────────────────────────────
def register_split(dataset_id: int,
                   n_train: int, n_val: int, n_test: int,
                   seed: int = 42,
                   train_ratio: float = 0.70,
                   val_ratio: float = 0.10) -> int:
    """
    Registra los parámetros de un split y devuelve split_id.
    Puedes llamar a esta función varias veces con distintos seeds
    para generar múltiples splits del mismo dataset.

    Ejemplo:
        split_a = register_split(dataset_id, 756, 84, 360, seed=42)
        split_b = register_split(dataset_id, 756, 84, 360, seed=123)
        split_c = register_split(dataset_id, 756, 84, 360, seed=999)
    """
    sql = """
        INSERT INTO Split
            (dataset_id, seed, train_ratio, val_ratio, n_train, n_val, n_test)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (dataset_id, seed, train_ratio, val_ratio,
                          n_train, n_val, n_test))
        return cur.lastrowid

def register_split_server(dataset_name: str,
                   server_name: int,
                   model_path : str,
                   n_train: int, n_val: int, n_test: int,
                   seed: int = 42,
                   train_ratio: float = 0.70,
                   val_ratio: float = 0.10) -> int:
    """
    Registra los parámetros de un split y devuelve split_id.
    Puedes llamar a esta función varias veces con distintos seeds
    para generar múltiples splits del mismo dataset.

    Ejemplo:
        split_a = register_split_server("rimone","server_rimone", "./rimone/model1/",756, 84, 360, seed=42_)

        split_b = register_split_server("rimone","server_rimone", "./rimone/model2/",756, 84, 360, seed=123)
    """
    sql = """
        INSERT INTO Split (dataset_id,server_id,model_path,seed,train_ratio,val_ratio,n_train,n_val,n_test) 
        SELECT Dataset.dataset_id, Server.server_id, %s, %s, %s, %s, %s, %s, %s 
        FROM Dataset,Server WHERE Dataset.name=%s and Server.name=%s ;
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (model_path, seed, train_ratio, val_ratio,
                          n_train, n_val, n_test,dataset_name,server_name))
        return cur.lastrowid




# ─────────────────────────────────────────────────────────────
# 3. Experiment
# ─────────────────────────────────────────────────────────────
def register_experiment(eve_model_path : str,
                        lr: float,
                        batch_size: int,
                        epochs_max: int,
                        patience: int,
                        description: str = None) -> int:
    """
    Crea un experimento con sus hiperparámetros y devuelve experiment_id.
    Llama a esta función ANTES de entrenar.

    Ejemplo:
        exp_id = register_experiment(
            lr=0.001, batch_size=32, epochs_max=500, patience=50,
            description="baseline ResNet50",
        )
    """
    sql = """
        INSERT INTO Experiment
            (description, lr, batch_size, epochs_max, patience, eve_model_path)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (description, lr, batch_size, epochs_max, patience, eve_model_path))
        return cur.lastrowid


# ─────────────────────────────────────────────────────────────
# 4. ExperimentSplit
# ─────────────────────────────────────────────────────────────
def register_experiment_split(experiment_id: int, split_id: int) -> int:
    """
    Vincula un experimento con un split y devuelve es_id.
    result_id queda a NULL hasta que el entrenamiento termine.

    Ejemplo — cruzar un experimento con tres splits:
        es_a = register_experiment_split(exp_id, split_a)
        es_b = register_experiment_split(exp_id, split_b)
        es_c = register_experiment_split(exp_id, split_c)
    """
    sql = """
        INSERT INTO ExperimentSplit (experiment_id, split_id)
        VALUES (%s, %s)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (experiment_id, split_id))
        return cur.lastrowid


def link_result_to_es(es_id: int, result_id: int) -> None:
    """
    Una vez terminado el entrenamiento, actualiza ExperimentSplit
    con el result_id correspondiente.

    Ejemplo:
        link_result_to_es(es_a, result_id)
    """
    sql = "UPDATE ExperimentSplit SET result_id = %s WHERE es_id = %s"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (result_id, es_id))


# ─────────────────────────────────────────────────────────────
# 5. TrainingResult
# ─────────────────────────────────────────────────────────────
def register_training_result(model_path: str,
                              best_epoch: int,
                              best_val_bal_acc: float,
                              test_bal_acc: float,
                              test_precision: float,
                              test_recall: float) -> int:
    """
    Registra el resultado final del entrenamiento y devuelve result_id.
    Llama a link_result_to_es() justo después para cerrar el ciclo.

    Ejemplo:
        result_id = register_training_result(
            model_path       = "best_resnet50_split42.pth",
            best_epoch       = 47,
            best_val_bal_acc = 0.8823,
            test_bal_acc     = bal_acc,
            test_precision   = precision,
            test_recall      = recall,
        )
        link_result_to_es(es_id, result_id)
    """
    sql = """
        INSERT INTO TrainingResult
            (model_path, best_epoch, best_val_bal_acc,
             test_bal_acc, test_precision, test_recall)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (model_path, best_epoch, best_val_bal_acc,
                          test_bal_acc, test_precision, test_recall))
        return cur.lastrowid


# ─────────────────────────────────────────────────────────────
# 5b. Federated round result persistence
# ─────────────────────────────────────────────────────────────

def save_federated_round_results(
        model_list: list,
        split_info_list: list,
        epochs_completed: int,
        aggregated_model_path: str,
) -> int:
    """
    Persists the outcome of one federated training round and links it to
    every ExperimentSplit that participated in the round.

    One TrainingResult row is created for the round. All splits that trained
    together share that single result because they produced one averaged model.
    Metrics are the weighted average across servers, where the weight of each
    server is proportional to its number of training samples.

    Parameters
    ----------
    model_list : list of (model_path, result_dict)
        Output collected from each server after the last epoch.
        result_dict must contain at minimum:
            train_samples   int
            val_samples     int   (used for weighting val metrics)
            val_bal_acc     float | None
            test_bal_acc    float | None
            test_precision  float | None
            test_recall     float | None
    split_info_list : list of dict
        One entry per server, in the same order as model_list.
        Each dict must contain 'es_id' (from get_experiment_splits).
    epochs_completed : int
        Total federated epochs run in this round (stored as best_epoch).
    aggregated_model_path : str
        Path to the averaged model produced by fl_compute_model().
        Stored as model_path in TrainingResult.

    Returns
    -------
    result_id : int
        The id of the newly created TrainingResult row.

    Example
    -------
        result_id = save_federated_round_results(
            model_list           = model_list,
            split_info_list      = split_info_list,
            epochs_completed     = cfg_experiment["epochs_max"],
            aggregated_model_path = cfg_experiment["eve_model_path"],
        )
    """
    if not model_list:
        raise ValueError("model_list is empty — nothing to save")
    if len(model_list) != len(split_info_list):
        raise ValueError(
            f"model_list length ({len(model_list)}) does not match "
            f"split_info_list length ({len(split_info_list)})"
        )

    # ── Weighted average of metrics ──────────────────────────────────────
    # Weight each server by its number of training samples.
    # If train_samples is missing or zero for all servers, fall back to
    # a uniform average so the function never crashes on partial results.
    raw_weights = [float((r.get("train_samples") or 0)) for (_, r) in model_list]
    total = sum(raw_weights)
    if total > 0:
        weights = [w / total for w in raw_weights]
    else:
        weights = [1.0 / len(model_list)] * len(model_list)

    def _wavg(key: str) -> float:
        """Return the weighted average of a metric across all servers."""
        return sum(
            wi * float(r.get(key) or 0.0)
            for wi, (_, r) in zip(weights, model_list)
        )

    avg_val_bal_acc    = _wavg("val_bal_acc")
    avg_test_bal_acc   = _wavg("test_bal_acc")
    avg_test_precision = _wavg("test_precision")
    avg_test_recall    = _wavg("test_recall")

    # ── Persist one shared TrainingResult ────────────────────────────────
    result_id = register_training_result(
        model_path       = aggregated_model_path,
        best_epoch       = epochs_completed,
        best_val_bal_acc = avg_val_bal_acc,
        test_bal_acc     = avg_test_bal_acc,
        test_precision   = avg_test_precision,
        test_recall      = avg_test_recall,
    )

    # ── Link result to every ExperimentSplit in this round ───────────────
    for split_info in split_info_list:
        es_id = split_info["es_id"]
        link_result_to_es(es_id, result_id)

    return result_id


# ─────────────────────────────────────────────────────────────
# 6. AdversarialRun
# ─────────────────────────────────────────────────────────────
def register_adversarial_run(result_id: int,
                              attack_type: str,
                              epsilon: float,
                              max_iter: int,
                              clean_bal_acc: float,
                              adv_bal_acc: float,
                              clean_precision: float,
                              adv_precision: float,
                              clean_recall: float,
                              adv_recall: float,
                              eps_step: float = None,
                              num_random_init: int = 1,
                              n_samples: int = None) -> int:
    """
    Registra los resultados de un ataque adversario y devuelve adv_id.

    Ejemplo — al final de run_attack() en adversarial_attacks.py:
        adv_id = register_adversarial_run(
            result_id       = result_id,
            attack_type     = "fgsm",
            epsilon         = 0.02,
            max_iter        = 10,
            clean_bal_acc   = clean_bal,   adv_bal_acc   = adv_bal,
            clean_precision = clean_prec,  adv_precision = adv_prec,
            clean_recall    = clean_rec,   adv_recall    = adv_rec,
        )
    """
    sql = """
        INSERT INTO AdversarialRun
            (result_id, attack_type, epsilon, eps_step, max_iter, num_random_init,
             n_samples, clean_bal_acc, adv_bal_acc,
             clean_precision, adv_precision,
             clean_recall, adv_recall)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (
            result_id, attack_type, epsilon, eps_step, max_iter, num_random_init,
            n_samples, clean_bal_acc, adv_bal_acc,
            clean_precision, adv_precision,
            clean_recall, adv_recall,
        ))
        return cur.lastrowid

# ─────────────────────────────────────────────────────────────
# 7. PoisoningRun
# ─────────────────────────────────────────────────────────────
def register_poisoning_run(result_id: int,
                            trigger_type: str,
                            percent_poison: float,
                            n_poisoned: int,
                            source_class: int,
                            target_class: int,
                            clean_bal_acc: float,
                            clean_precision: float,
                            clean_recall: float,
                            attack_success_rate: float,
                            ac_precision: float,
                            ac_recall: float,
                            ac_f1: float,
                            trigger_size: int = None,
                            trigger_position: str = None) -> int:
    """
    Registra un ataque de backdoor (entrenamiento envenenado) junto con su
    detección por Activation Clustering, y devuelve poison_id.

    trigger_size y trigger_position son opcionales: NULL en el trigger
    'sinusoidal' (que es global y no tiene parche localizado).
    """
    sql = """
        INSERT INTO PoisoningRun
            (result_id, trigger_type, trigger_size, trigger_position,
             percent_poison, n_poisoned, source_class, target_class,
             clean_bal_acc, clean_precision, clean_recall,
             attack_success_rate,
             ac_precision, ac_recall, ac_f1)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (
            result_id, trigger_type, trigger_size, trigger_position,
            percent_poison, n_poisoned, source_class, target_class,
            clean_bal_acc, clean_precision, clean_recall,
            attack_success_rate,
            ac_precision, ac_recall, ac_f1,
        ))
        return cur.lastrowid


# ─────────────────────────────────────────────────────────────
# 7. MembershipInferenceRun
# ─────────────────────────────────────────────────────────────
def register_mia_run(result_id: int,
                     attack_variant: str,
                     n_train_samples: int,
                     n_test_samples: int,
                     mia_accuracy: float,
                     mia_precision: float,
                     mia_recall: float) -> int:
    """
    Registra los resultados de un Membership Inference Attack y devuelve mia_id.

    Ejemplo — al final de run_mia() en membership_inference.py:
        mia_id = register_mia_run(
            result_id       = result_id,
            attack_variant  = "rf",
            n_train_samples = metrics["n_train_samples"],
            n_test_samples  = metrics["n_test_samples"],
            mia_accuracy    = metrics["mia_accuracy"],
            mia_precision   = metrics["mia_precision"],
            mia_recall      = metrics["mia_recall"],
        )
    """
    sql = """
        INSERT INTO MembershipInferenceRun
            (result_id, attack_variant,
             n_train_samples, n_test_samples,
             mia_accuracy, mia_precision, mia_recall)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (
            result_id, attack_variant,
            n_train_samples, n_test_samples,
            mia_accuracy, mia_precision, mia_recall,
        ))
        return cur.lastrowid
    
    
# ─────────────────────────────────────────────────────────────
# Utilidades de consulta
# ─────────────────────────────────────────────────────────────
# Result lookup helpers
# ─────────────────────────────────────────────────────────────

def get_result_info(result_id: int) -> dict:
    """
    Returns the TrainingResult row for a given result_id as a dict.
    Keys: result_id, model_path, best_epoch, best_val_bal_acc,
          test_bal_acc, test_precision, test_recall.
    Raises ValueError if the result_id does not exist.

    Example:
        info = get_result_info(7)
        model_path = info["model_path"]
    """
    sql = "SELECT * FROM TrainingResult WHERE result_id = %s"
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, (result_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"No TrainingResult found with result_id={result_id}. "
            "Run new_experiment.py first to generate a training result."
        )
    return dict(row)


def get_latest_result_id() -> int:
    """
    Returns the result_id of the most recently inserted TrainingResult row.
    Useful as a fallback when --result_id is not provided on the CLI.
    Raises ValueError if the TrainingResult table is empty.

    Example:
        result_id = get_latest_result_id()
        info      = get_result_info(result_id)
    """
    sql = "SELECT result_id FROM TrainingResult ORDER BY result_id DESC LIMIT 1"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            "TrainingResult table is empty. "
            "Run new_experiment.py first to generate a training result."
        )
    return row[0]


# ─────────────────────────────────────────────────────────────
def get_all_results() -> list[dict]:
    """
    Devuelve todos los resultados de entrenamiento con su contexto
    (dataset, split, hiperparámetros), ordenados por test_bal_acc desc.
    Útil para comparar qué combinación experimento+split funcionó mejor.
    """
    sql = """
        SELECT
            e.experiment_id,
            e.description,
            d.name              AS dataset,
            s.seed              AS split_seed,
            s.split_id,
            e.lr,
            e.batch_size,
            e.epochs_max,
            e.patience,
            r.best_epoch,
            r.best_val_bal_acc,
            r.test_bal_acc,
            r.test_precision,
            r.test_recall,
            r.model_path
        FROM ExperimentSplit es
        JOIN Experiment     e  ON es.experiment_id = e.experiment_id
        JOIN Split          s  ON es.split_id      = s.split_id
        JOIN Dataset        d  ON s.dataset_id     = d.dataset_id
        JOIN TrainingResult r  ON es.result_id     = r.result_id
        ORDER BY r.test_bal_acc DESC
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        return cur.fetchall()


def get_results_by_experiment(experiment_id: int) -> list[dict]:
    """
    Devuelve todos los resultados de un experimento concreto,
    uno por cada split sobre el que se ejecutó.
    Permite ver la varianza del modelo entre splits distintos.
    """
    sql = """
        SELECT
            s.seed              AS split_seed,
            s.n_train, s.n_val, s.n_test,
            r.best_epoch,
            r.best_val_bal_acc,
            r.test_bal_acc,
            r.test_precision,
            r.test_recall,
            r.model_path
        FROM ExperimentSplit es
        JOIN Split          s  ON es.split_id  = s.split_id
        JOIN TrainingResult r  ON es.result_id = r.result_id
        WHERE es.experiment_id = %s
        ORDER BY r.test_bal_acc DESC
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, (experiment_id,))
        return cur.fetchall()


def get_adversarial_summary() -> list[dict]:
    """
    Devuelve todos los ataques adversarios con las métricas
    limpias y adversarias, incluyendo el experimento y split atacado.
    """
    sql = """
        SELECT
            a.adv_id,
            e.experiment_id,
            e.description       AS experiment,
            d.name              AS dataset,
            s.seed              AS split_seed,
            a.attack_type,
            a.epsilon,
            a.max_iter,
            a.n_samples,
            a.clean_bal_acc,
            a.adv_bal_acc,
            ROUND(a.clean_bal_acc - a.adv_bal_acc, 4) AS bal_acc_drop,
            a.clean_precision,
            a.adv_precision,
            a.clean_recall,
            a.adv_recall
        FROM AdversarialRun  a
        JOIN TrainingResult  r  ON a.result_id     = r.result_id
        JOIN ExperimentSplit es ON es.result_id     = r.result_id
        JOIN Experiment      e  ON es.experiment_id = e.experiment_id
        JOIN Split           s  ON es.split_id      = s.split_id
        JOIN Dataset         d  ON s.dataset_id     = d.dataset_id
        ORDER BY a.adv_id DESC
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        return cur.fetchall()


def get_poisoning_summary() -> list[dict]:
    """
    Devuelve todos los ataques de backdoor con su contexto, ordenados por
    ASR descendente. Útil para identificar qué triggers funcionaron mejor
    contra qué modelos y qué tan efectiva fue la defensa AC.
    """
    sql = """
        SELECT
            p.poison_id,
            e.experiment_id,
            e.description       AS experiment,
            d.name              AS dataset,
            s.seed              AS split_seed,
            p.trigger_type,
            p.trigger_size,
            p.trigger_position,
            p.percent_poison,
            p.n_poisoned,
            p.source_class,
            p.target_class,
            p.clean_bal_acc,
            p.attack_success_rate,
            p.ac_precision,
            p.ac_recall,
            p.ac_f1
        FROM PoisoningRun    p
        JOIN TrainingResult  r  ON p.result_id     = r.result_id
        JOIN ExperimentSplit es ON es.result_id    = r.result_id
        JOIN Experiment      e  ON es.experiment_id = e.experiment_id
        JOIN Split           s  ON es.split_id      = s.split_id
        JOIN Dataset         d  ON s.dataset_id     = d.dataset_id
        ORDER BY p.attack_success_rate DESC
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        return cur.fetchall()


def get_mia_summary() -> list[dict]:
    """
    Devuelve todos los ataques MIA con su contexto (experimento, dataset,
    split), ordenados por mia_accuracy desc. Útil para identificar
    rápidamente qué combinaciones presentan mayor fuga de privacidad.
    """
    sql = """
        SELECT
            m.mia_id,
            e.experiment_id,
            e.description       AS experiment,
            d.name              AS dataset,
            s.seed              AS split_seed,
            m.attack_variant,
            m.n_train_samples,
            m.n_test_samples,
            m.mia_accuracy,
            m.mia_precision,
            m.mia_recall,
            r.test_bal_acc
        FROM MembershipInferenceRun m
        JOIN TrainingResult  r  ON m.result_id      = r.result_id
        JOIN ExperimentSplit es ON es.result_id     = r.result_id
        JOIN Experiment      e  ON es.experiment_id = e.experiment_id
        JOIN Split           s  ON es.split_id      = s.split_id
        JOIN Dataset         d  ON s.dataset_id     = d.dataset_id
        ORDER BY m.mia_accuracy DESC
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        return cur.fetchall()

# ─────────────────────────────────────────────────────────────
# A1. Server
# ─────────────────────────────────────────────────────────────
def register_server(name: str, owner_email: str, owner_password: str) -> int:
    """
    Inserta un servidor y devuelve su server_id.
    Si el nombre ya existe devuelve el id existente (idempotente).

    Ejemplo:
        server_id = register_server(
            name="server_rimone",
            owner_email="test1@codigla.org",
            owner_password="changethis"
        )
    """
    sql = """
        INSERT INTO Server (name, owner_client_email, owner_client_password)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE server_id = LAST_INSERT_ID(server_id)
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, (name, owner_email, owner_password))
        return cur.lastrowid

# ------------------------------------------------------------------------
# A2. Get dataset_path
# --------------------------------------------------------------------
def get_dataset_path_by_name(name: str) -> str:
    """
    Devuelve dataset_id de la tabla de datasets a partir del nombre
    
    Ejemplo:
        dataset_id=get_dataset_by_name(name)

    """

    sql = """"
        SELECT dataset_id FROM Dataset WHERE name=%s
        """

    with get_db() as conn:
        cur=conn.cursor()
        cur.execute(sql,(name,))
        row = cur.fetchone()
        if row is not None:
            return row["path"]
        else:
            return ""

# -----------------------------------------------
# A3. Get experiment parameters
# ------------------------------------------------


def get_experiment(experiment_id : int) -> dict:
    """

    Deuelve la informaci'on del experimento a partir de su id

    """

    sql = """
            SELECT * FROM Experiment WHERE experiment_id = %s

          """

    with get_db() as conn:
        cur=conn.cursor()
        cur.execute(sql,(experiment_id,))
        row=cur.fetchone()
        column_names=[f[0] for f in cur.description]
        data=dict((field,row[n]) for (n,field) in enumerate(column_names))
        return data


# --------------------------------------------------
# A4. Get splits of experiment
# --------------------------------------------------

def get_experiment_splits(experiment_id : int) -> dict:
    """
    Returns the list of splits for an experiment, grouped by server_id.

    Each value in the returned dict is a list of split dicts ordered by
    split_id. Each dict contains all Split columns plus es_id (from
    ExperimentSplit), which is required to call link_result_to_es() after
    a federated training round completes.

    Return structure:
        {
            "3": [ {split_id, es_id, dataset_id, seed, ...}, ... ],
            "7": [ {split_id, es_id, dataset_id, seed, ...}, ... ],
        }
    """
    # Explicit column list avoids ambiguous split_id from the JOIN and
    # guarantees es_id is always present in every row dict.
    sql = """
        SELECT
            s.split_id,
            s.dataset_id,
            s.server_id,
            s.seed,
            s.train_ratio,
            s.val_ratio,
            s.n_train,
            s.n_val,
            s.n_test,
            s.model_path,
            s.epochs,
            es.es_id,
            es.result_id
        FROM Split s
        INNER JOIN ExperimentSplit es ON es.split_id = s.split_id
        WHERE es.experiment_id = %s
        ORDER BY s.split_id ASC
    """
    with get_db() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, (experiment_id,))
        rows = cur.fetchall()

    splits = {}
    for row in rows:
        server_id = str(row["server_id"])
        if server_id not in splits:
            splits[server_id] = []
        splits[server_id].append(dict(row))
    return splits

# -----------------------------------
# A5. Get server info using id

def get_server(server_id : int) -> dict:
    """

    Deuelve la informaci'on del servidor a partir de su id

    """

    sql = """
            SELECT * FROM Server WHERE server_id = %s

          """

    with get_db() as conn:
        cur=conn.cursor()
        cur.execute(sql,(server_id,))
        row=cur.fetchone()
        column_names=[f[0] for f in cur.description]
        data=dict((field,row[n]) for (n,field) in enumerate(column_names))
        return data

# ---------------------------------------------
# A6. Get dataset info using id

def get_dataset(dataset_id : int) -> dict:
    """

    Deuelve la informaci'on del dataset a partir de su id

    """

    sql = """
            SELECT * FROM Dataset WHERE dataset_id = %s

          """

    with get_db() as conn:
        cur=conn.cursor()
        cur.execute(sql,(dataset_id,))
        row=cur.fetchone()
        column_names=[f[0] for f in cur.description]
        data=dict((field,row[n]) for (n,field) in enumerate(column_names))
        return data