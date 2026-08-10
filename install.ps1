<#
.SYNOPSIS
    Creates the "pdf2html" conda environment (name kept for backward
    compatibility with existing installs; runs doc2html.py) and installs
    dependencies.
.PARAMETER WithBlip
    Also install the optional offline BLIP captioning backend
    (requirements-blip.txt). Not needed for --generate-alt lmstudio or
    plain conversion without --generate-alt.
.EXAMPLE
    .\install.ps1
    .\install.ps1 -WithBlip
#>
param(
    [switch]$WithBlip
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param([string]$Description, [scriptblock]$Action)
    Write-Host $Description
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Шаг завершился с ошибкой: $Description"
    }
}

Invoke-Step "Creating conda environment `"pdf2html`" (python 3.11)..." {
    conda create -y -n pdf2html python=3.11
}

Invoke-Step "Upgrading pip..." {
    conda run -n pdf2html python -m pip install --upgrade pip
}

Invoke-Step "Installing dependencies (requirements.txt)..." {
    conda run -n pdf2html python -m pip install -r requirements.txt
}

if ($WithBlip) {
    Invoke-Step "Installing optional BLIP captioning backend (requirements-blip.txt)..." {
        conda run -n pdf2html python -m pip install -r requirements-blip.txt
    }
}

Write-Host ""
Write-Host "Done. Run '. .\activate.ps1' (dot-sourced) to activate the environment."

if (-not $WithBlip) {
    Write-Host ""
    Write-Host "Note: the offline BLIP captioning backend (--generate-alt blip) is NOT"
    Write-Host "installed by default. Either use --generate-alt lmstudio, or install it"
    Write-Host "later with:"
    Write-Host "  conda run -n pdf2html python -m pip install -r requirements-blip.txt"
    Write-Host "Or rerun this script as: .\install.ps1 -WithBlip"
}
