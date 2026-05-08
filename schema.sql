-- ─────────────────────────────────────────────────────────────
-- Base de datos: federate  (v2)
-- Ejecutar como: mysql -u tfg0 -p < schema.sql
-- ─────────────────────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS federate
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE federate;

-- ─────────────────────────────────────────────────────────────
-- 1. Dataset
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Dataset (
    dataset_id      INT             NOT NULL AUTO_INCREMENT,
    name            VARCHAR(100)    NOT NULL,
    path            VARCHAR(500)    NOT NULL,
    total_samples   INT             NOT NULL,
    samples_class0  INT             NOT NULL,
    samples_class1  INT             NOT NULL,
    PRIMARY KEY (dataset_id),
    UNIQUE KEY uq_dataset_name (name)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────
-- 2. Split
--    Parámetros de división train/val/test sobre un dataset.
--    seed + ratios permiten reproducir exactamente los mismos
--    índices en cualquier momento.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Split (
    split_id        INT             NOT NULL AUTO_INCREMENT,
    dataset_id      INT             NOT NULL,
    seed            INT             NOT NULL DEFAULT 42,
    train_ratio     FLOAT           NOT NULL DEFAULT 0.70,
    val_ratio       FLOAT           NOT NULL DEFAULT 0.10,
    n_train         INT             NOT NULL,
    n_val           INT             NOT NULL,
    n_test          INT             NOT NULL,
    PRIMARY KEY (split_id),
    CONSTRAINT fk_split_dataset
        FOREIGN KEY (dataset_id) REFERENCES Dataset(dataset_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────
-- 3. Experiment
--    Configuración de hiperparámetros independiente del split.
--    Un mismo experimento puede correr sobre varios splits.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Experiment (
    experiment_id   INT             NOT NULL AUTO_INCREMENT,
    description     VARCHAR(300)    DEFAULT NULL,
    lr              FLOAT           NOT NULL,
    batch_size      INT             NOT NULL,
    epochs_max      INT             NOT NULL,
    patience        INT             NOT NULL,
    PRIMARY KEY (experiment_id)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────
-- 4. TrainingResult
--    Resultado final para una combinación experimento+split.
--    Se crea antes de ExperimentSplit para evitar referencia
--    circular en los FKs.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS TrainingResult (
    result_id           INT             NOT NULL AUTO_INCREMENT,
    model_path          VARCHAR(500)    NOT NULL,
    best_epoch          INT             NOT NULL,
    best_val_bal_acc    FLOAT           NOT NULL,
    test_bal_acc        FLOAT           NOT NULL,
    test_precision      FLOAT           NOT NULL,
    test_recall         FLOAT           NOT NULL,
    PRIMARY KEY (result_id)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────
-- 5. ExperimentSplit
--    Tabla pivote: relaciona cada experimento con cada split
--    sobre el que se ha ejecutado.
--    result_id es NULL hasta que el entrenamiento termina.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ExperimentSplit (
    es_id           INT             NOT NULL AUTO_INCREMENT,
    experiment_id   INT             NOT NULL,
    split_id        INT             NOT NULL,
    result_id       INT             DEFAULT NULL,
    PRIMARY KEY (es_id),
    UNIQUE KEY uq_exp_split (experiment_id, split_id),
    CONSTRAINT fk_es_experiment
        FOREIGN KEY (experiment_id) REFERENCES Experiment(experiment_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_es_split
        FOREIGN KEY (split_id) REFERENCES Split(split_id)
        ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT fk_es_result
        FOREIGN KEY (result_id) REFERENCES TrainingResult(result_id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────
-- 6. AdversarialRun
--    Resultado de un ataque sobre un modelo entrenado concreto.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AdversarialRun (
    adv_id                  INT             NOT NULL AUTO_INCREMENT,
    result_id               INT             NOT NULL,
    attack_type             VARCHAR(10)     NOT NULL,
    epsilon                 FLOAT           NOT NULL,
    eps_step                FLOAT           DEFAULT NULL,
    max_iter                INT             NOT NULL DEFAULT 10,
    num_random_init         INT             NOT NULL DEFAULT 1,
    n_samples               INT             DEFAULT NULL,
    clean_bal_acc           FLOAT           NOT NULL,
    adv_bal_acc             FLOAT           NOT NULL,
    clean_precision         FLOAT           NOT NULL,
    adv_precision           FLOAT           NOT NULL,
    clean_recall            FLOAT           NOT NULL,
    adv_recall              FLOAT           NOT NULL,
    PRIMARY KEY (adv_id),
    CONSTRAINT fk_adv_result
        FOREIGN KEY (result_id) REFERENCES TrainingResult(result_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;


CREATE TABLE IF NOT EXISTS PoisoningRun (
    poison_id           INT NOT NULL AUTO_INCREMENT,
    result_id           INT NOT NULL,

    -- ── Ataque dirty-label ──────────────────────────────────
    trigger_type        VARCHAR(20) NOT NULL,    -- 'square' | 'cross' | 'checkerboard' | 'gaussian' | 'sinusoidal' | 'border'
    trigger_size        INT,                     -- NULL en sinusoidal
    trigger_position    VARCHAR(20),             -- NULL en sinusoidal
    percent_poison      FLOAT       NOT NULL,
    n_poisoned          INT         NOT NULL,
    source_class        INT         NOT NULL,
    target_class        INT         NOT NULL,

    -- ── Métricas del modelo entrenado ───────────────────────
    clean_bal_acc       FLOAT       NOT NULL,
    clean_precision     FLOAT       NOT NULL,
    clean_recall        FLOAT       NOT NULL,
    attack_success_rate FLOAT       NOT NULL,

    -- ── Detección con Activation Clustering ─────────────────
    ac_precision        FLOAT       NOT NULL,
    ac_recall           FLOAT       NOT NULL,
    ac_f1               FLOAT       NOT NULL,

    PRIMARY KEY (poison_id),
    CONSTRAINT fk_poison_result
        FOREIGN KEY (result_id) REFERENCES TrainingResult(result_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;


CREATE TABLE IF NOT EXISTS MembershipInferenceRun (
    mia_id          INT   NOT NULL AUTO_INCREMENT,
    result_id       INT   NOT NULL,
    attack_variant  VARCHAR(30) NOT NULL,  -- 'rule_based', 'rf', 'nn', ...
    n_train_samples INT   NOT NULL,
    n_test_samples  INT   NOT NULL,
    mia_accuracy    FLOAT NOT NULL,        -- métrica principal
    mia_precision   FLOAT NOT NULL,
    mia_recall      FLOAT NOT NULL,
    PRIMARY KEY (mia_id),
    CONSTRAINT fk_mia_result
        FOREIGN KEY (result_id) REFERENCES TrainingResult(result_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;