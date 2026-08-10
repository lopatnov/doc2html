<#
.SYNOPSIS
    Activates the "pdf2html" conda environment (used by doc2html.py; name
    kept for backward compatibility with existing installs) in the current
    shell.
.DESCRIPTION
    Must be dot-sourced so the environment activation (PATH, prompt, etc.)
    persists in your interactive session rather than only inside this
    script's own scope:
        . .\activate.ps1
    Running it as ".\activate.ps1" (without the leading ". ") will not
    leave the environment active afterward.

    Requires PowerShell to already be conda-initialized - if `conda
    activate` fails with something like "CommandNotFoundError" or "run
    'conda init' before 'conda activate'", run once (new PowerShell
    session afterward):
        conda init powershell
#>

Write-Host "Activating pdf2html environment..."
conda activate pdf2html
