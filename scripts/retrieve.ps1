<#
.SYNOPSIS
    Run one hybrid retrieval query (M2) and print the ranked hits.

.DESCRIPTION
    PowerShell equivalent of `make retrieve-fake` for Windows machines without
    make. It is a thin wrapper: the job itself is
    `python -m production_rag.retrieval`, owned by the retrieval package, and
    every flag below maps to one of its options.

    What this does: embeds the question (dense), encodes it as BM25 term weights
    (sparse), queries both named vectors in the target Qdrant collection, fuses
    the two ranked lists with reciprocal rank fusion, and prints the top hits.

    What this does NOT do: rerank (M3), generate an answer, or produce citations
    (M4). There is no LLM call anywhere in this path. The output is retrieved
    passages, not an answer.

    Two execution modes:

      -In container (default) runs inside the `api` service, so QDRANT_URL
       resolves to the compose hostname and no host-side install is needed.
       Requires the stack to be up (`.\scripts\up.ps1`).

      -OnHost runs the module from the active interpreter against
       http://localhost:6333. Faster edit loop, needs `pip install -e ".[rag]"`.

    The embedder must be the SAME one that built the collection. A collection
    ingested with `fake` and queried with `openai` compares two unrelated vector
    spaces: it returns hits, ranked by nothing. Nothing detects this for you --
    both vectors have 1536 dimensions.

    Requires a collection ingested by M2. A collection left over from M1 has no
    `sparse` named vector and hybrid retrieval aborts against it; rebuild with
    `.\scripts\ingest.ps1 -Recreate`.

.PARAMETER Query
    The question, as a user would phrase it. Required.

.PARAMETER TopK
    Hits to print after fusion. Defaults to the job's configured
    retrieval.top_k (12).

.PARAMETER Mode
    hybrid | dense | sparse. Default hybrid. `dense` and `sparse` query one
    branch only, which is how a regression gets attributed to a branch.

.PARAMETER Embedder
    fake | openai. Default fake. Must match the collection.

.PARAMETER Collection
    Target Qdrant collection. Defaults to the job's configured collection.

.PARAMETER ConfigPath
    YAML config to load. Default configs/default.yaml

.PARAMETER Json
    Emit only the JSON result object, no human-readable table.

.PARAMETER OnHost
    Run against the host interpreter instead of the api container.

.EXAMPLE
    .\scripts\retrieve.ps1 -Query "How does reciprocal rank fusion work?"
    .\scripts\retrieve.ps1 -Query "QDRANT__SERVICE__GRPC_PORT" -Mode sparse
    .\scripts\retrieve.ps1 -Query "what is a cross-encoder" -TopK 5 -OnHost
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Query,
    [int]$TopK,
    [ValidateSet('hybrid', 'dense', 'sparse')]
    [string]$Mode = 'hybrid',
    [ValidateSet('fake', 'openai')]
    [string]$Embedder = 'fake',
    [string]$Collection,
    [string]$ConfigPath = 'configs/default.yaml',
    [switch]$Json,
    [switch]$OnHost
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (-not $Query.Trim()) { throw 'query is empty.' }

    $jobArgs = @(
        '-m', 'production_rag.retrieval',
        '--config', $ConfigPath,
        '--query', $Query,
        '--mode', $Mode,
        '--embedder', $Embedder
    )
    if ($TopK) { $jobArgs += @('--top-k', "$TopK") }
    if ($Collection) { $jobArgs += @('--collection', $Collection) }
    if ($Json) { $jobArgs += '--json' }

    if ($Embedder -eq 'openai' -and -not $env:OPENAI_API_KEY -and -not (Test-Path '.env')) {
        # Not fatal: Compose may inject it from elsewhere. Say so now rather
        # than after the query embed call 401s.
        Write-Host "warning: OPENAI_API_KEY not set and no .env present." -ForegroundColor Yellow
    }

    if ($OnHost) {
        if (-not $Json) { Write-Host "==> python $($jobArgs -join ' ')" -ForegroundColor Cyan }
        & python @jobArgs
    }
    else {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "docker not found on PATH. Use -OnHost, or install Docker Desktop."
        }
        $composeArgs = @('compose', 'run', '--rm', 'api', 'python') + $jobArgs
        if (-not $Json) { Write-Host "==> docker $($composeArgs -join ' ')" -ForegroundColor Cyan }
        & docker @composeArgs
    }

    if ($LASTEXITCODE -ne 0) {
        # Exit 2 is a bad invocation or a collection that cannot serve this mode
        # (an M1 collection has no `sparse` vector); retrying will not fix it.
        if ($LASTEXITCODE -eq 2) {
            Write-Host ""
            Write-Host "if the collection predates M2, rebuild it:" -ForegroundColor Yellow
            Write-Host "  .\scripts\ingest.ps1 -Recreate"
        }
        throw "retrieve failed with exit code $LASTEXITCODE"
    }

    if (-not $Json) {
        Write-Host ""
        Write-Host "these are retrieved passages, not an answer." -ForegroundColor DarkGray
        Write-Host "generation with citations is M4." -ForegroundColor DarkGray
    }
}
finally {
    Pop-Location
}
