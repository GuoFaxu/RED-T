# RED-T

RED-T contains two modules:

- `rice-rediff/`: sequence generation with REDiff.
- `reap/`: enhancer activity prediction with REAP.

## Workflow

```powershell
powershell -ExecutionPolicy Bypass -File workflow_redt.ps1 -NumSamples 500
```

The workflow writes generated sequences to:

```text
rice-rediff/sequences/rediff_epoch2400_gen500.fasta
```

REAP prediction outputs are written under:

```text
reap/results/
```

Required model files:

```text
rice-rediff/checkpoints/rediff_epoch2400.pth
reap/models/pretrained/plant-dnabert-6mer/
reap/models/finetuned/fold1/merged/
reap/models/finetuned/fold2/merged/
reap/models/finetuned/fold3/merged/
reap/models/finetuned/fold4/merged/
reap/models/finetuned/fold5/merged/
```
