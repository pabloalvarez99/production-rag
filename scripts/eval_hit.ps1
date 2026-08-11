<#
.SYNOPSIS
    Score source-level hit@k over the golden set with the M2 retriever.

.DESCRIPTION
    PowerShell equivalent of `make eval-hit-fake` for Windows machines without
    make. It is a thin wrapper over `scripts/eval_hit.py`, which is stdlib-only
    and carries the actual logic and its caveats.

    This reports one coarse metric and gates nothing. It asks, per golden item,
    whether any retrieved chunk came from a labelled document. It says nothing
    about answer quality: there is no generation in M2.

    Read the number with the embedder that produced it. On `fake` the dense
    branch is hash noise, so the score is a plumbing assertion -- with one real
    exception, since BM25 weights come from the text and the sparse branch is
    genuinely lexical even there. On `openai` it is a measurement over 14 items,
    which is a smoke test with error bars a whole document wide.

    Requires a collection ingested by M2 (`.\scripts\ingest.ps1 -Recreate` if it
    predates M2) and, in container mode, a running stack.

.PARAMETER Embedder
    fake | openai. Default fake. Must match the embedder that built the
    collection: nothing detects a mismatch, both produce 1536 dimensions.

.PARAMETER PerBranch
    Also score dense-only and sparse-only runs. Triples the retrievals, and on
    -Embedder openai triples the embedding spend.

.PARAMETER K
    k values to report. Default 1, 3, 5, 10.

.PARAMETER Collection
    Target Qdrant collection. Defaults to the job's configured collection.

.PARAMETER ConfigPath
    YAML config to load. Default configs/default.yaml

.PARAMETER Json
    Emit only the JSON summary.

.PARAMETER OnHost
    Run against the host interpreter instead of the api container. Default is
    the container, where QDRANT_URL resolves to the compose hostname.

.EXAMPLE
    .\scripts\eval_hit.ps1
    .\scripts\eval_hit.ps1 -PerBranch
    .\scripts\eval_hit.ps1 -Embedder openai -K 1,5
#>
[CmdletBinding()]
param(
    [ValidateSet('fake', 'openai')]
    [string]$Embedder = 'fake',
    [switch]$PerBranch,
    [int[]]$K = @(1, 3, 5, 10),
    [string]$Collection,
    [string]$ConfigPath = 'configs/default.yaml',
    [switch]$Json,
    [switch]$OnHost
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    $jobArgs = @(
        'scripts/eval_hit.py',
        '--config', $ConfigPath,
        '--embedder', $Embedder,
        '--k'
    ) + ($K | ForEach-Object { "$_" })
    if ($Collection) { $jobArgs += @('--collection', $Collection) }
    if ($PerBranch) { $jobArgs += '--per-branch' }
    if ($Json) { $jobArgs += '--json' }

    if ($OnHost) {
        if (-not $Json) { Write-Host "==> python $($jobArgs -join ' ')" -ForegroundColor Cyan }
        & python @jobArgs
    }
    else {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "docker not found on PATH. Use -OnHost, or install Docker Desktop."
        }
        # scripts/ is excluded from the image (see .dockerignore), so mount it.
        $composeArgs = @(
            'compose', 'run', '--rm',
            '-v', "$($repoRoot -replace '\\', '/')/scripts:/app/scripts:ro",
            'api', 'python'
        ) + $jobArgs
        if (-not $Json) { Write-Host "==> docker $($composeArgs -join ' ')" -ForegroundColor Cyan }
        & docker @composeArgs
    }

    if ($LASTEXITCODE -ne 0) { throw "eval failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
