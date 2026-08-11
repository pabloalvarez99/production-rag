<#
.SYNOPSIS
    Run both M6 evaluation tiers through the unified runner.

.DESCRIPTION
    PowerShell equivalent of `make eval-all-fake`. Fake providers are the safe
    default: no key, network, or spend. They validate plumbing and structural
    metrics only; use real providers deliberately for quality measurements.

    Container mode mounts data/eval/reports writable because the rest of data/
    is read-only in Compose. Use -OnHost for the active Python environment.

.EXAMPLE
    .\scripts\eval_all.ps1
    .\scripts\eval_all.ps1 -Sample 10 -Seed 42
#>
[CmdletBinding()]
param(
    [ValidateSet('fake', 'openai')]
    [string]$Embedder = 'fake',
    [ValidateSet('fake', 'openai')]
    [string]$Llm = 'fake',
    [ValidateRange(1, 1000)]
    [int]$K = 5,
    [ValidateRange(1, 1000000)]
    [int]$Sample,
    [int]$Seed = 42,
    [ValidateRange(0.0, 1.0)]
    [double]$FailUnderHit = 0.0,
    [string]$ReportPath = 'data/eval/reports/all-fake.json',
    [switch]$OnHost
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$reportsPath = Join-Path $repoRoot 'data/eval/reports'
Push-Location $repoRoot

try {
    $jobArgs = @(
        '-m', 'production_rag.evals.run',
        '--tier', 'all',
        '--embedder', $Embedder,
        '--llm', $Llm,
        '--k', "$K",
        '--seed', "$Seed",
        '--fail-under-hit', "$FailUnderHit",
        '--report', $ReportPath
    )
    if ($PSBoundParameters.ContainsKey('Sample')) { $jobArgs += @('--sample', "$Sample") }

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

    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
