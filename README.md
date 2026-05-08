# Federated Learning for Medical Data — Robustness Against Adversarial Threats

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![PySyft](https://img.shields.io/badge/PySyft-federated-orange)
![ART](https://img.shields.io/badge/ART-Adversarial%20Robustness%20Toolbox-purple)
![Status](https://img.shields.io/badge/status-research%20%2F%20TFG-lightgrey)

Medical imaging is a domain where two pressures pull in opposite directions: hospitals cannot freely share patient data, which makes **federated learning** an attractive paradigm — models travel, data does not — but the very mechanics that make it viable (multiple participants, remote code execution on private datasets, model exchange) also widen the attack surface considerably. A malicious client can implant a backdoor during training, an adversary with API access can craft inputs that flip a diagnosis, and a curious party can try to recover whether a specific patient was part of the training set.

This repository is the experimental framework of a Bachelor's Thesis (*Trabajo de Fin de Grado*) at the **Universidad de La Laguna** that quantifies how exposed a realistic glaucoma classifier — a ResNet-50 trained on fundus images — is to each of these threats. It implements three families of attacks (evasion via FGSM and PGD, data poisoning through six backdoor trigger families with Activation Clustering as defence, and membership inference for privacy leakage), runs them in both centralised and federated (PySyft) settings, and tracks every experiment in a MySQL database so that comparisons across triggers, poisoning rates and attack types can be reproduced from a single SQL query.

## Installation

Python ≥ 3.10 and a CUDA-capable GPU recommended. The project is installable via `pyproject.toml`:

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Additionally:

- **PyTorch / torchvision** matching your CUDA version (install separately).
- **MySQL ≥ 8.0**, initialised with `mysql -u <user> -p < schema.sql`. Connection parameters are read by `database_access.py`.

## Usage

**1. Register a dataset.** Extracts class-labelled zips into a unified layout and inserts a row in `Dataset`:

```bash
python create_dataset.py \
    --dataset_dir ./data/glaucoma_dataset \
    --dataset_name GLAUCOMA \
    --zips normal.zip glaucoma.zip \
    --classes normal glaucoma
```

**2. Clean training + adversarial evaluation.** `run_pipeline.py` trains a federated ResNet-50 baseline and immediately runs FGSM/PGD against the resulting checkpoint, persisting both the `TrainingResult` and the corresponding `AdversarialRun` rows:

```bash
python run_pipeline.py \
    --data_dir ./data/glaucoma_dataset \
    --save_path resnet50_clean.pth
```

**3. Backdoor attack + defence.** `run_backdoor_pipeline.py` chains training with a poisoned dataset, Activation Clustering detection and DB persistence (`PoisoningRun`). Six trigger families are supported: `square`, `cross`, `checkerboard`, `gaussian`, `sinusoidal`, `border`.

```bash
python run_backdoor_pipeline.py \
    --data_dir ./data/glaucoma_dataset \
    --trigger_type square --trigger_size 8 --trigger_pos top_left \
    --poison_rate 0.2 --source_class 1 --target_class 0 \
    --save_path backdoor_square.pth
```

**4. Membership inference.** `run_mia_pipeline.py` runs a black-box MIA against a trained checkpoint and writes a `MembershipInferenceRun` row:

```bash
python run_mia_pipeline.py \
    --data_dir ./data/glaucoma_dataset \
    --model_path resnet50_clean.pth \
    --attack_variant rule_based
```

> Each underlying stage (`train_resnet.py`, `adversarial_attacks.py`, `backdoor_attack.py`, `backdoor_defense.py`, `membership_inference.py`) can also be invoked standalone if you want to skip the database integration.

## Reproducibility

Every pipeline is deterministic for a given seed: dataset splits are reproduced from `(dataset_id, seed, train_ratio, val_ratio)`, trigger patterns are seeded, and trained checkpoints are versioned by their `result_id`. Comparative analyses across triggers, poison rates and attack types are produced by SQL queries directly on the `federate` database.

## Author

**Rubén Díaz Marrero** — Grado en Ingeniería Informática, Universidad de La Laguna. Trabajo de Fin de Grado, curso 2025/2026. Tutor: **José Ignacio Estévez Damas**.