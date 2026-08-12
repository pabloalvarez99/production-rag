<# Start the credential-free, recordable demo stack. #>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not (Test-Path 'data/corpus')) { throw 'data/corpus is missing; integrate the wave 8 corpus first.' }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'docker is required.' }

    $env:QDRANT_COLLECTION = 'prag_demo'
    & docker compose up -d qdrant
    if ($LASTEXITCODE -ne 0) { throw 'Qdrant failed to start.' }

    $deadline = (Get-Date).AddSeconds(120)
    do {
        try { $ready = (Invoke-WebRequest 'http://localhost:6333/readyz' -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 }
        catch { $ready = $false }
        if (-not $ready) { Start-Sleep -Seconds 2 }
    } while (-not $ready -and (Get-Date) -lt $deadline)
    if (-not $ready) { throw 'Qdrant did not become ready within 120 seconds.' }

    & docker compose build api
    if ($LASTEXITCODE -ne 0) { throw 'The demo API image failed to build.' }

    & docker compose run --rm api python -c "from importlib.metadata import version; import httpx; client=version('qdrant-client'); server=httpx.get('http://qdrant:6333/').json()['version']; compatible=lambda value: tuple(value.split('.')[:2]); assert compatible(client) == compatible(server), f'Qdrant major.minor mismatch: client={client} server={server}'"
    if ($LASTEXITCODE -ne 0) { throw 'Qdrant client/server major.minor versions do not match.' }

    & docker compose run --rm api python -m production_rag.ingest --config configs/default.yaml --source data/corpus --embedder fake --collection prag_demo --recreate-collection
    if ($LASTEXITCODE -ne 0) { throw 'Offline ingest failed.' }

    & docker compose up -d api
    if ($LASTEXITCODE -ne 0) { throw 'The demo server failed to start.' }

    Write-Host ''
    Write-Host 'Demo ready: http://localhost:8000/' -ForegroundColor Green
    Write-Host '1. Why does hybrid search use reciprocal rank fusion?'
    Write-Host '2. Who won the Antarctic underwater chess championship?'
}
finally {
    Pop-Location
}
