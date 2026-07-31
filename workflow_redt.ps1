param(
    [int]$NumSamples = 500,
    [string]$Output = "rice-rediff/sequences/rediff_epoch2400_gen500.fasta",
    [string]$Device = "auto",
    [string]$Folds = "auto",
    [string]$BaseModel = "reap/models/pretrained/plant-dnabert-6mer",
    [string]$ModelsRoot = "reap/models/finetuned",
    [string]$ResultsRoot = "reap/results"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Join-RedTRoot {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Path))
}

$OutputPath = Join-RedTRoot $Output
$BaseModelPath = Join-RedTRoot $BaseModel
$ModelsRootPath = Join-RedTRoot $ModelsRoot
$ResultsRootPath = Join-RedTRoot $ResultsRoot

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null

Push-Location (Join-Path $Root "rice-rediff")
try {
    & python scripts/sample_rediff.py `
        --config configs/rediff_rice_512.yaml `
        --checkpoint checkpoints/rediff_epoch2400.pth `
        --num-samples $NumSamples `
        --output $OutputPath `
        --device $Device
}
finally {
    Pop-Location
}

Push-Location (Join-Path $Root "reap")
try {
    & python scripts/predict_reap.py `
        --input $OutputPath `
        --base_model $BaseModelPath `
        --models_root $ModelsRootPath `
        --results_root $ResultsRootPath `
        --folds $Folds `
        --suffix "rediff_gen$NumSamples"
}
finally {
    Pop-Location
}
