# RAPF

Implementation of **Reliability-aware prototype fusion for activity recognition with arbitrary missing sensors.**

## Install

Use Python 3.11 in an isolated environment. From the repository root:

```bash
python -m pip install -e .
```

## Data

Download [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). `python -m rapf.ucihar --root "data/UCI HAR Dataset" --output data/raw.npz` imports body acceleration, gyroscope and total acceleration as three modalities. PAMAP2/Opportunity can use the same contract: `x: (N,M,C,T)`, `y`, `group`, `split: train/val/test`, optional `available: (N,M)` and `sample_missing: (N,M)`. Pad smaller channel groups consistently. Whole subjects remain in one split; every sample retains at least one modality.

## Workflow

```bash
python -m rapf.stress --input data/raw.npz --output data/missing30.npz --missing-rate 0.3
python -m rapf prepare --input data/missing30.npz
python -m rapf train
python -m rapf run
python -m rapf evaluate
python -m rapf plot
```

Main commands accept `--config config.json` and `--output runs/default`. Use a separate output directory for each seed and condition. Its `settings.json` records the resolved configuration.

## Configuration

`missing_rate` controls training masks. `temperature` and `prototype_temperature` are separate reliability/prototype temperatures. `prototype_weight` weights prototype CE. `clip_threshold` defines a standardized quality descriptor. `classes` and `width` define task size and shared embeddings. Validation selects the checkpoint using `patience`. Common training fields `epochs`, `batch_size`, `learning_rate`, `width` and `seed` control optimization, model size and randomness where present. Inspect `config.json` for defaults.

## Outputs

`predictions.npz` stores probabilities, modality weights and availability. `metrics.csv` includes accuracy, Macro-F1, weighted-F1, balanced accuracy, NLL, ECE, selective risk, parameter count and host latency. `risk_coverage.csv` and its PNG describe retained-set classification error.

## Method and comparisons

Available content and absent-sensor tokens use separate branches, ensuring missing identities are not multiplied away. Temporal CNN encoders share a common output dimension. Reliability uses encoded content, variance, clipping and sample missingness. Complete and masked views train together. Prototypes and representations are L2-normalized before their distance loss.

Controls: `full`, `uniform`, `no-tokens`, `no-prototype`, `early`, `late`. Select a control with `--variant` and keep the same data and seed. Changes to trainable objectives or architectures require a separate training run.

The `rapf.compare` command averages repeated rows within independent units, then reports means, bootstrap intervals, paired Wilcoxon comparisons, rank-biserial effects and Holm adjustment per metric. For example:

```bash
python -m rapf.compare --inputs "runs/*/metrics.csv" --reference full --metrics accuracy macro_f1 --unit seed
```

