# REAP

REAP predicts rice enhancer activity from 512 bp DNA sequences using Plant DNABERT-6mer.

## Repository Structure

- `src/reap/`: model, training, inference, sequence processing, metrics, and I/O utilities.
- `scripts/train_reap_5fold.py`: training entry point.
- `scripts/predict_reap.py`: inference entry point.
- `configs/reap_512.yaml`: run configuration.
- `examples/example_sequences.fasta`: example inference input.
- `results/extras/global_scaler_20251121_093312.json`: target scaling parameters.

## Installation

```bash
conda env create -f environment.yml
conda activate rice-reap
pip install -e .
```

## Data

Training expects:

```text
data/processed/all_data.csv
data/processed/test_data.csv
```

Required columns:

```text
chromosome,sequence,expression
```

## Training

```bash
python scripts/train_reap_5fold.py \
  --data-dir data/processed \
  --pretrained models/pretrained/plant-dnabert-6mer \
  --out-root models \
  --results-root results
```

## Inference

```bash
python scripts/predict_reap.py \
  --input examples/example_sequences.fasta \
  --base_model models/pretrained/plant-dnabert-6mer \
  --models_root models/finetuned \
  --results_root results \
  --folds auto
```
