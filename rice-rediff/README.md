# Rice REDiff

REDiff is the diffusion-model component used to generate synthetic 512 bp rice enhancer sequences in the RED-T study. This repository is a REDiff-only core-code release: it covers data preparation, diffusion training, and sequence sampling. Other RED-T components and downstream analyses are not included here.

## Repository Contents

- `src/rediff/`: REDiff model code, including the U-Net denoiser and Gaussian diffusion implementation.
- `scripts/prepare_data.py`: clean sequences, normalize them to 512 bp, and reproduce the train/validation split.
- `scripts/train_rediff.py`: train REDiff from a CSV of rice enhancer sequences.
- `scripts/sample_rediff.py`: sample FASTA sequences from a trained REDiff checkpoint.
- `configs/rediff_rice_512.yaml`: default configuration matching the archived 512 bp rice enhancer run.
- `checkpoints/rediff_epoch2400.pth`: REDiff checkpoint path used for 500-sequence generation.

## Installation

Recommended GPU installation:

```bash
conda env create -f environment.yml
conda activate rice-rediff
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The `environment.yml` uses Python 3.10.13, PyTorch 2.1.1, and CUDA 12.1. If CUDA is available, the last command should print `True`.

## Data Preparation

If `all_data.csv` is available, prepare deterministic 512 bp train/validation FASTA files with:

```bash
python scripts/prepare_data.py \
  --input ../data/all_data.csv \
  --out-dir data/processed \
  --sequence-col sequence \
  --seq-len 512 \
  --val-frac 0.10 \
  --seed 42
```

For a tiny smoke test, use:

```bash
python scripts/prepare_data.py --input examples/toy_sequences.csv --out-dir outputs/toy_data --val-frac 0.25
```

## Training REDiff

Full training is GPU-intensive. The archived run used 13,414 rice enhancer sequences, 3,000 epochs, batch size 512 with gradient accumulation, and 512 bp one-hot sequence tensors shaped as `[4, 512, 1]`.

```bash
python scripts/train_rediff.py \
  --config configs/rediff_rice_512.yaml \
  --data ../data/all_data.csv \
  --out-dir outputs/rediff_3000
```

The script saves:

- `outputs/rediff_3000/checkpoints/*best-model.pt`
- `outputs/rediff_3000/checkpoints/*best-ema.pt`
- `outputs/rediff_3000/checkpoints/*epoch-XXXX-model.pt`
- `outputs/rediff_3000/loss_curve.csv`

## Sampling

Sample sequences from a trained checkpoint:

```bash
python scripts/sample_rediff.py \
  --config configs/rediff_rice_512.yaml \
  --checkpoint checkpoints/rediff_epoch2400.pth \
  --num-samples 500 \
  --output sequences/rediff_epoch2400_gen500.fasta
```
