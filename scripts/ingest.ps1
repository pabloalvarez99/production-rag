<#
.SYNOPSIS
    Run the M1 ingest job over a corpus directory and report what landed in Qdrant.

.DESCRIPTION
    PowerShell equivalent of `make ingest-fake` / `make ingest-sample` for
    Windows machines without make. It is a thin wrapper: the job itself is
    `python -m production_rag.ingest`, owned by the ingest package, and every
    flag below maps to one of its options.

    Two execution modes:

      -In container (default) runs inside the `api` service, so QDRANT_URL
       resolves to the compose hostname and no host-side install is needed.
       Requires the stack to be up (`.\scripts\up.ps1`).

      -OnHost runs the module from the active interpreter against
       http://localhost:6333. Faster edit loop, needs `pip install -e ".[rag]"`.

    The embedder choice is the important one. `fake` is a deterministic hash
    embedder: no API key, no network, no cost, and the same text always maps to
    the same vector. It makes the whole ingest path -- walk, chunk, embed,
    upsert, count -- runnable in CI and on a laptop with no credentials. The
    vectors are meaningless for retrieval quality, which is the point: it tests
    the plumbing and claims nothing about relevance.

    `openai` reads OPENAI_API_KEY from the environment (or from the gitignored
    .env that Compose loads). The key is never passed on the command line, where
    it would land in shell history and in `docker inspect`.

.PARAMETER Source
    Corpus *root*, relative to the repo root. Payload `source_path` values are
    relative to it and its first path segment becomes the filterable `source`
    field, so data/raw is what makes a sample chunk read as
    `sample/08-bm25-vs-dense.md` -- exactly what data/eval/golden.jsonl labels.
    Pointing this deeper stops those labels matching. Default data/raw

.PARAMETER Embedder
    fake | openai. Default fake.

.PARAMETER Collection
    Target Qdrant collection. Defaults to the job's configured collection.

.PARAMETER ConfigPath
    YAML config to load. Default configs/default.yaml

.PARAMETER Recreate
    Drop and recreate the collection before upserting. Destructive: every vector
    in that collection is lost and must be re-ingested.

.PARAMETER DryRun
    Walk and chunk, report the counts, embed and upsert nothing.

.PARAMETER OnHost
    Run against the host interpreter instead of the api container.

.EXAMPLE
    .\scripts\ingest.ps1
    .\scripts\ingest.ps1 -DryRun
    .\scripts\ingest.ps1 -Embedder openai
    .\scripts\ingest.ps1 -OnHost -Recreate
#>
[CmdletBinding()]
param(
    [string]$Source = 'data/raw',
    [ValidateSet('fake', 'openai')]
    [string]$Embedder = 'fake',
    [string]$Collection,
    [string]$ConfigPath = 'configs/default.yaml',
    [switch]$Recreate,
    [switch]$DryRun,
    [switch]$OnHost
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (-not (Test-Path (Join-Path $repoRoot $Source))) {
        throw "corpus directory not found: $Source"
    }

    $jobArgs = @(
        '-m', 'production_rag.ingest',
        '--config', $ConfigPath,
        '--source', $Source,
        '--embedder', $Embedder
    )
    if ($Collection) { $jobArgs += @('--collection', $Collection) }
    if ($Recreate) { $jobArgs += '--recreate' }
    if ($DryRun) { $jobArgs += '--dry-run' }

    if ($Embedder -eq 'openai' -and -not $env:OPENAI_API_KEY -and -not (Test-Path '.env')) {
        # Not fatal: Compose may inject it from elsewhere. Say so now rather
        # than after the walk has finished and the first embed call 401s.
        Write-Host "warning: OPENAI_API_KEY not set and no .env present." -ForegroundColor Yellow
    }

    if ($Recreate) {
        Write-Host "==> -Recreate will DROP the target collection before ingest." -ForegroundColor Yellow
        $answer = Read-Host "type 'recreate' to continue"
        if ($answer -ne 'recreate') { throw 'aborted by user.' }
    }

    if ($OnHost) {
        Write-Host "==> python $($jobArgs -join ' ')" -ForegroundColor Cyan
        & python @jobArgs
    }
    else {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "docker not found on PATH. Use -OnHost, or install Docker Desktop."
        }
        $composeArgs = @('compose', 'run', '--rm', 'api', 'python') + $jobArgs
        Write-Host "==> docker $($composeArgs -join ' ')" -ForegroundColor Cyan
        & docker @composeArgs
    }

    if ($LASTEXITCODE -ne 0) { throw "ingest failed with exit code $LASTEXITCODE" }

    Write-Host ""
    Write-Host "ingest finished" -ForegroundColor Green
    Write-Host "  verify   .\scripts\health.ps1"
    Write-Host "  browse   http://localhost:6333/dashboard"
}
finally {
    Pop-Location
}
