<#
.SYNOPSIS
    Run the free deterministic M6 retrieval evaluation tier.

.DESCRIPTION
    PowerShell equivalent of `make eval-tier1`. It calls A1's unified runner
    without reimplementing metrics. The default fake embedder needs no key and
    proves the evaluation path is wired; its dense score is not a quality claim.

    Container mode mounts data/eval/reports writable because the rest of data/
    is read-only in Compose. Use -OnHost for the active Python environment.

.EXAMPLE
    .\scripts\eval_tier1.ps1
    .\scripts\eval_tier1.ps1 -K 5 -FailUnderHit 0.8
#>
[CmdletBinding()]
param(
    [ValidateSet('fake', 'openai')]
    [string]$Embedder = 'fake',
    [ValidateRange(1, 1000)]
    [int]$K = 5,
    [ValidateRange(0.0, 1.0)]
    [double]$FailUnderHit = 0.0,
    [string]$ReportPath = 'data/eval/reports/tier1.json',
    [switch]$OnHost
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$reportsPath = Join-Path $repoRoot 'data/eval/reports'
Push-Location $repoRoot

try {
    $jobArgs = @(
        '-m', 'production_rag.evals.run',
        '--tier', '1',
        '--embedder', $Embedder,
        '--k', "$K",
        '--fail-under-hit', "$FailUnderHit",
        '--report', $ReportPath
    )

    if ($OnHost) {
        Write-Host "==> python $($jobArgs -join ' ')" -ForegroundColor Cyan
        & python @jobArgs
    }
    else {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw 'docker not found on PATH. Use -OnHost, or install Docker Desktop.'
        }
        $mount = "$($reportsPath -replace '\\', '/'):/app/data/eval/reports"
        $composeArgs = @('compose', 'run', '--rm', '-v', $mount, 'api', 'python') + $jobArgs
        Write-Host "==> docker $($composeArgs -join ' ')" -ForegroundColor Cyan
        & docker @composeArgs
    }

    if ($LASTEXITCODE -ne 0) { throw "Tier 1 eval failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
