# RAPF

Implementation of **Reliability-aware prototype fusion for activity recognition with arbitrary missing sensors.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Download [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). The importer uses body acceleration, gyroscope and total acceleration as three modalities. PAMAP2/Opportunity can use the same contract: `x: (N,M,C,T)`, `y`, `group`, `split: train/val/test`, optional `available: (N,M)` and `sample_missing: (N,M)`. Pad smaller channel groups consistently. Whole subjects remain in one split; every sample retains at least one modality.

```bash
python -m rapf.ucihar --root "data/UCI HAR Dataset" --output data/raw.npz
python -m rapf.stress --input data/raw.npz --output data/missing30.npz --missing-rate 0.3
python -m rapf prepare --input data/missing30.npz --config config.json --output runs/full
```

## Training and inference

```bash
python -m rapf train --config config.json --output runs/full
python -m rapf run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m rapf evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `uniform`, `no-tokens`, `no-prototype`, `early`, `late`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m rapf prepare --input data/missing30.npz --config config.json --output runs/uniform --variant uniform
python -m rapf train --config config.json --output runs/uniform --variant uniform
python -m rapf run --config config.json --output runs/uniform --variant uniform
python -m rapf evaluate --config config.json --output runs/uniform --variant uniform
```

## Outputs

All main outputs are written below `runs/full/`.

`predictions.npz` stores probabilities, modality weights and availability. `metrics.csv` includes accuracy, Macro-F1, weighted-F1, balanced accuracy, NLL, ECE, selective risk, parameter count and host latency. `risk_coverage.csv` and its PNG describe retained-set classification error.

```bash
python -m rapf plot --config config.json --output runs/full
```
