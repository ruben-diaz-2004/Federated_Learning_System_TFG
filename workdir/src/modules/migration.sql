-- =============================================================================
-- Migration: experiment_type support
-- =============================================================================
-- Changes:
--   1. Add `name`            VARCHAR(100) UNIQUE  to Experiment
--   2. Add `experiment_type` VARCHAR(30)  NOT NULL to Experiment
--   3. Add AdversarialExperimentParams  (attack hyperparams stored at creation)
--   4. Add MiaExperimentParams          (attack hyperparams stored at creation)
--   5. Add BackdoorExperimentParams     (attack hyperparams stored at creation)
--
-- Allowed values for experiment_type:
--   'training'   - federated training via new_experiment.py
--   'adversarial'- adversarial attack  via run_adversarial_experiment.py
--   'mia'        - membership inference via run_mia_experiment.py
--   'backdoor'   - backdoor attack      via run_backdoor_experiment.py
--
-- Backward compatibility:
--   Existing rows receive experiment_type = 'training' and a generated name.
-- =============================================================================

-- 1. Add columns to Experiment
-- -----------------------------------------------------------------------------
ALTER TABLE `Experiment`
    ADD COLUMN `name` VARCHAR(100) NULL
        COMMENT 'Short unique label for the experiment (e.g. adv_pgd_refuge_s42)'
        AFTER `experiment_id`,
    ADD COLUMN `experiment_type` VARCHAR(30) NOT NULL DEFAULT 'training'
        COMMENT 'training | adversarial | mia | backdoor'
        AFTER `name`,
    ADD UNIQUE KEY `uq_experiment_name` (`name`);

-- Back-fill name for pre-existing rows so the UNIQUE constraint is satisfied.
UPDATE `Experiment`
SET `name` = CONCAT('experiment_', `experiment_id`)
WHERE `name` IS NULL;

-- 2. AdversarialExperimentParams
-- -----------------------------------------------------------------------------
-- Stores the attack hyperparameters chosen when the adversarial experiment is
-- registered. One row per Experiment of type 'adversarial'.
DROP TABLE IF EXISTS `AdversarialExperimentParams`;
CREATE TABLE `AdversarialExperimentParams` (
    `adv_exp_id`        INT NOT NULL AUTO_INCREMENT,
    `experiment_id`     INT NOT NULL,
    `attack_types`      VARCHAR(100) NOT NULL DEFAULT 'fgsm,pgd,bim'
        COMMENT 'Comma-separated list of attack types to run',
    `epsilon`           FLOAT NOT NULL DEFAULT 0.1,
    `eps_step`          FLOAT NULL
        COMMENT 'Step size for PGD/BIM; NULL means epsilon/4',
    `max_iter`          INT NOT NULL DEFAULT 10,
    `num_random_init`   INT NOT NULL DEFAULT 1,
    `n_samples`         INT NULL
        COMMENT 'Max samples per attack; NULL means all',
    `batch_size`        INT NOT NULL DEFAULT 32,
    PRIMARY KEY (`adv_exp_id`),
    UNIQUE KEY `uq_adv_exp` (`experiment_id`),
    CONSTRAINT `fk_adv_exp_experiment`
        FOREIGN KEY (`experiment_id`) REFERENCES `Experiment` (`experiment_id`)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3
  COMMENT='Hyperparameters for adversarial attack experiments';

-- 3. MiaExperimentParams
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS `MiaExperimentParams`;
CREATE TABLE `MiaExperimentParams` (
    `mia_exp_id`        INT NOT NULL AUTO_INCREMENT,
    `experiment_id`     INT NOT NULL,
    `variants`          VARCHAR(100) NOT NULL DEFAULT 'all'
        COMMENT 'Comma-separated MIA variants or "all"',
    `test_size`         FLOAT NOT NULL DEFAULT 0.5
        COMMENT 'Fraction of shadow data used as MIA test set',
    `n_shadow_samples`  INT NULL
        COMMENT 'Number of shadow samples; NULL means use all available',
    PRIMARY KEY (`mia_exp_id`),
    UNIQUE KEY `uq_mia_exp` (`experiment_id`),
    CONSTRAINT `fk_mia_exp_experiment`
        FOREIGN KEY (`experiment_id`) REFERENCES `Experiment` (`experiment_id`)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3
  COMMENT='Hyperparameters for membership inference attack experiments';

-- 4. BackdoorExperimentParams
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS `BackdoorExperimentParams`;
CREATE TABLE `BackdoorExperimentParams` (
    `bd_exp_id`         INT NOT NULL AUTO_INCREMENT,
    `experiment_id`     INT NOT NULL,
    `trigger_type`      VARCHAR(20) NOT NULL DEFAULT 'square',
    `trigger_size`      INT NULL,
    `trigger_position`  VARCHAR(20) NULL,
    `percent_poison`    FLOAT NOT NULL DEFAULT 0.1,
    `source_class`      INT NOT NULL DEFAULT 0,
    `target_class`      INT NOT NULL DEFAULT 1,
    `check_all_classes` TINYINT(1) NOT NULL DEFAULT 0,
    PRIMARY KEY (`bd_exp_id`),
    UNIQUE KEY `uq_bd_exp` (`experiment_id`),
    CONSTRAINT `fk_bd_exp_experiment`
        FOREIGN KEY (`experiment_id`) REFERENCES `Experiment` (`experiment_id`)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3
  COMMENT='Hyperparameters for backdoor attack experiments';