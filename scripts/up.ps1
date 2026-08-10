<#
.SYNOPSIS
    Build and start the production-rag stack, then wait until it is healthy.

.DESCRIPTION
    PowerShell equivalent of `make up` for Windows machines without make.
    Exits non-zero if the API does not report healthy within -TimeoutSeconds,
    so it is safe to chain in CI or in a wrapper script.

.PARAMETER NoBuild
    Skip the image rebuild and start whatever image already exists.

.PARAMETER TimeoutSeconds
    How long to wait for the API healthcheck to pass. Default 120.

.EXAMPLE
    .\scripts\up.ps1
    .\scripts\up.ps1 -NoBuild -TimeoutSeconds 60
#>
[CmdletBinding()]
param(
    [switch]$NoBuild,
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "docker not found on PATH. Install Docker Desktop and retry."
    }

    $composeArgs = @('compose', 'up', '-d')
    if (-not $NoBuild) { $composeArgs += '--build' }

    Write-Host "==> docker $($composeArgs -join ' ')" -ForegroundColor Cyan
    & docker @composeArgs
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed with exit code $LASTEXITCODE" }

    Write-Host "==> waiting for the API to become healthy (timeout ${TimeoutSeconds}s)" -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $healthy = $false

    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) { $healthy = $true; break }
        }
        catch {
            # Connection refused while uvicorn boots is expected; keep polling.
        }
        Start-Sleep -Seconds 3
    }

    if (-not $healthy) {
        Write-Host "==> API did not become healthy in time. Last 50 log lines:" -ForegroundColor Red
        & docker compose logs --tail=50 api
        throw "API healthcheck did not pass within ${TimeoutSeconds}s."
    }

    Write-Host ""
    Write-Host "stack is up" -ForegroundColor Green
    Write-Host "  API docs        http://localhost:8000/docs"
    Write-Host "  API health      http://localhost:8000/health"
    Write-Host "  Qdrant console  http://localhost:6333/dashboard"
}
finally {
    Pop-Location
}
