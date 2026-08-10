<#
.SYNOPSIS
    Stop the production-rag stack.

.DESCRIPTION
    PowerShell equivalent of `make down`. By default the Qdrant storage volume
    is preserved, so a later `up.ps1` finds the collection already ingested.
    Pass -Purge to delete it -- that is destructive and requires re-ingest.

.PARAMETER Purge
    Also remove named volumes (drops the vector index) and orphan containers.

.EXAMPLE
    .\scripts\down.ps1
    .\scripts\down.ps1 -Purge
#>
[CmdletBinding()]
param(
    [switch]$Purge
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    $composeArgs = @('compose', 'down')

    if ($Purge) {
        Write-Host "WARNING: -Purge deletes the qdrant_storage volume." -ForegroundColor Yellow
        Write-Host "         All ingested vectors are lost and must be re-ingested." -ForegroundColor Yellow
        $answer = Read-Host "Type 'yes' to continue"
        if ($answer -ne 'yes') {
            Write-Host "aborted; nothing was removed." -ForegroundColor Yellow
            return
        }
        $composeArgs += @('-v', '--remove-orphans')
    }

    Write-Host "==> docker $($composeArgs -join ' ')" -ForegroundColor Cyan
    & docker @composeArgs
    if ($LASTEXITCODE -ne 0) { throw "docker compose down failed with exit code $LASTEXITCODE" }

    Write-Host "stack is down" -ForegroundColor Green
}
finally {
    Pop-Location
}
