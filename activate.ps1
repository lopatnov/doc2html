<#
.SYNOPSIS
    Activates the "doc2html" conda environment in the current shell.
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

Write-Host "Activating doc2html environment..."
conda activate doc2html
