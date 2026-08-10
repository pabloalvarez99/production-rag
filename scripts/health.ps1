<#
.SYNOPSIS
    Probe every health surface of the running stack and print a verdict table.

.DESCRIPTION
    Checks, in order:
      1. Docker container state and health for both services.
      2. GET /health        on the API (liveness).
      3. GET /v1/health     on the API (versioned readiness, includes deps).
      4. GET /readyz        on Qdrant.
      5. GET /collections   on Qdrant (is the target collection present?).

    Exits 0 only when every required probe passes, so it works as a gate in
    scripts and CI. A missing collection is reported as a warning, not a
    failure -- an empty stack before ingest is a valid state.

.PARAMETER BaseUrl
    API base URL. Default http://localhost:8000

.PARAMETER QdrantUrl
    Qdrant base URL as seen from the host. Default http://localhost:6333

.PARAMETER Collection
    Collection name expected after ingest. Default production_rag

.EXAMPLE
    .\scripts\health.ps1
    .\scripts\health.ps1 -BaseUrl http://localhost:8080
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://localhost:8000',
    [string]$QdrantUrl = 'http://localhost:6333',
    [string]$Collection = 'production_rag'
)

$ErrorActionPreference = 'Continue'
$results = New-Object System.Collections.Generic.List[object]
$failed = $false

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Uri,
        [switch]$Required
    )

    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 10 -UseBasicParsing
        $sw.Stop()
        return [pscustomobject]@{
            Check    = $Name
            Status   = 'PASS'
            Code     = $response.StatusCode
            Ms       = [int]$sw.ElapsedMilliseconds
            Detail   = ($response.Content -replace '\s+', ' ').Trim()
            Required = [bool]$Required
        }
    }
    catch {
        return [pscustomobject]@{
            Check    = $Name
            Status   = if ($Required) { 'FAIL' } else { 'WARN' }
            Code     = ''
            Ms       = 0
            Detail   = $_.Exception.Message
            Required = [bool]$Required
        }
    }
}

Write-Host "==> container state" -ForegroundColor Cyan
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    Push-Location $repoRoot
    & docker compose ps
    Pop-Location
}
else {
    Write-Host "docker not on PATH; skipping container state." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==> endpoint probes" -ForegroundColor Cyan

$results.Add((Test-Endpoint -Name 'api /health'      -Uri "$BaseUrl/health"      -Required))
$results.Add((Test-Endpoint -Name 'api /v1/health'   -Uri "$BaseUrl/v1/health"   -Required))
$results.Add((Test-Endpoint -Name 'qdrant /readyz'   -Uri "$QdrantUrl/readyz"    -Required))
$results.Add((Test-Endpoint -Name 'qdrant /collections' -Uri "$QdrantUrl/collections"))

$results | Format-Table Check, Status, Code, Ms -AutoSize

foreach ($r in $results) {
    if ($r.Status -eq 'FAIL') {
        $failed = $true
        Write-Host "FAIL $($r.Check): $($r.Detail)" -ForegroundColor Red
    }
    elseif ($r.Status -eq 'WARN') {
        Write-Host "WARN $($r.Check): $($r.Detail)" -ForegroundColor Yellow
    }
}

# Collection presence is informational: before the first ingest it is absent
# and the stack is still correct.
$collectionsProbe = $results | Where-Object { $_.Check -eq 'qdrant /collections' }
if ($collectionsProbe.Status -eq 'PASS') {
    if ($collectionsProbe.Detail -match [regex]::Escape($Collection)) {
        Write-Host "collection '$Collection' is present." -ForegroundColor Green
    }
    else {
        Write-Host "collection '$Collection' not found yet -- run the ingest job." -ForegroundColor Yellow
    }
}

Write-Host ""
if ($failed) {
    Write-Host "health: FAIL" -ForegroundColor Red
    exit 1
}

Write-Host "health: OK" -ForegroundColor Green
exit 0
