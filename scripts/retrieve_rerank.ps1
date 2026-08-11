<#
.SYNOPSIS
    Run one retrieval query with the M3 cross-encoder rerank stage and print the
    ranked hits.

.DESCRIPTION
    Sibling of scripts/retrieve.ps1. That one is the M2 surface (retrieve and
    stop at fusion); this one adds the rerank flag and the comparison mode, so
    the M2 command keeps working exactly as documented and nothing about the
    default path changes.

    What this does: everything retrieve.ps1 does — embed the question (dense),
    encode it as BM25 term weights (sparse), query both named vectors, fuse the
    two ranked lists with reciprocal rank fusion — and then hands the fused
    candidates to a cross-encoder that rescores each (query, passage) pair and
    keeps the best few.

    What this does NOT do: generate an answer or produce citations (M4). There is
    no LLM call anywhere in this path. Reranking reorders passages; it does not
    answer.

    Why the stage exists: RRF orders by RANK, never magnitude, and neither branch
    ever reads the query and the passage together. A cross-encoder does, in one
    forward pass — which is what makes it accurate and what makes it impossible
    to index. Retrieval owns recall; rerank owns precision at the top. It never
    queries Qdrant, so it can only reorder what fusion already returned.

    HONEST SCOPE. -Rerank fake is a plumbing double: it scores by the share of
    query terms a passage contains, which is a cruder version of what BM25 did
    one stage earlier. It exists so the whole stage is exercisable with no
    credential, no download and no network. It is not a quality claim and no
    ordering it produces should ever be quoted. The providers that mean something
    are `local` (BAAI/bge-reranker-base on CPU) and `cohere` (hosted).

    Two execution modes, same as retrieve.ps1:

      -In container (default) runs inside the `api` service, so QDRANT_URL
       resolves to the compose hostname and no host-side install is needed.
       Requires the stack to be up (`.\scripts\up.ps1`).

      -OnHost runs the module from the active interpreter against
       http://localhost:6333. Faster edit loop, needs `pip install -e ".[rag]"`.

    The embedder must be the SAME one that built the collection, and a collection
    ingested by M2 or later is required — see retrieve.ps1 for both caveats; they
    are unchanged here.

.PARAMETER Query
    The question, as a user would phrase it. Required.

.PARAMETER Rerank
    off | fake | local | cohere | auto. Default `fake`, because it is the only
    value that always runs: no key, no model download, no network. It proves the
    stage is wired and nothing about relevance. Use `local` or `cohere` for a
    result worth reading, `off` for plain M2 behaviour, `auto` to obey
    rerank.enabled / rerank.provider from the YAML.

    `local` needs sentence-transformers and pulls ~1.1 GB of weights on first
    use. `cohere` needs the cohere package and COHERE_API_KEY in the environment
    (never as a flag: a key on a command line lands in shell history and in
    `docker inspect`).

.PARAMETER Compare
    Run the query twice — once with -Rerank off, once with the chosen provider —
    and print both. The only honest way to see what the stage did: an ordering
    on its own is not evidence, a delta is.

.PARAMETER TopK
    Hits to keep after reranking. Defaults to the job's configured rerank.top_k
    (6). Note this is a smaller number than the M2 retrieval.top_k (12): the
    reranker is fed more than it keeps, on purpose.

.PARAMETER Mode
    hybrid | dense | sparse. Default hybrid. Attributes a regression to a branch.

.PARAMETER Embedder
    fake | openai. Default fake. Must match the collection.

.PARAMETER Collection
    Target Qdrant collection. Defaults to the job's configured collection.

.PARAMETER ConfigPath
    YAML config to load. Default configs/default.yaml

.PARAMETER Json
    Suppress this wrapper's commentary so stdout carries only the job's output,
    whose last line is a single JSON object. Ignored with -Compare, which always
    prints both runs and their headers.

.PARAMETER OnHost
    Run against the host interpreter instead of the api container.

.EXAMPLE
    .\scripts\retrieve_rerank.ps1 -Query "how does reciprocal rank fusion work"
    .\scripts\retrieve_rerank.ps1 -Query "what is a cross-encoder" -Rerank local
    .\scripts\retrieve_rerank.ps1 -Query "what is a cross-encoder" -Rerank local -Compare
    .\scripts\retrieve_rerank.ps1 -Query "QDRANT__SERVICE__GRPC_PORT" -Rerank off -Mode sparse

.EXAMPLE
    # Did the stage run, over how many candidates, and did it fail open?
    .\scripts\retrieve_rerank.ps1 -Query "…" -Rerank local -Json |
        ConvertFrom-Json | Select-Object -ExpandProperty rerank

.EXAMPLE
    # What did it actually move? A pre_rerank_rank far below rank is the stage
    # earning its latency.
    .\scripts\retrieve_rerank.ps1 -Query "…" -Rerank local -Json |
        ConvertFrom-Json | Select-Object -ExpandProperty hits |
        Select-Object rank, pre_rerank_rank, rerank_score, source_path
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Query,
    [ValidateSet('off', 'fake', 'local', 'cohere', 'auto')]
    [string]$Rerank = 'fake',
    [switch]$Compare,
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

function Invoke-Retrieve {
    <#
        One run of the retrieve job. Returns the exit code; output goes straight
        to the console so a -Json caller can pipe it, exactly as retrieve.ps1
        behaves.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$RerankKind,
        [switch]$Quiet
    )

    $jobArgs = @(
        '-m', 'production_rag.retrieve',
        '--config', $ConfigPath,
        '--query', $Query,
        '--mode', $Mode,
        '--embedder', $Embedder,
        '--rerank', $RerankKind
    )
    if ($TopK) { $jobArgs += @('--top-k', "$TopK") }
    if ($Collection) { $jobArgs += @('--collection', $Collection) }

    # No --json flag is passed: the job already writes its logs to stderr and a
    # single JSON object as the last line of stdout. -Json here only suppresses
    # this wrapper's own commentary, so the stdout stream stays parseable.
    $announce = -not ($Json -and -not $Compare) -and -not $Quiet

    if ($OnHost) {
        if ($announce) { Write-Host "==> python $($jobArgs -join ' ')" -ForegroundColor Cyan }
        & python @jobArgs
    }
    else {
        $composeArgs = @('compose', 'run', '--rm', 'api', 'python') + $jobArgs
        if ($announce) { Write-Host "==> docker $($composeArgs -join ' ')" -ForegroundColor Cyan }
        & docker @composeArgs
    }
    return $LASTEXITCODE
}

try {
    if (-not $Query.Trim()) { throw 'query is empty.' }

    if (-not $OnHost -and -not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "docker not found on PATH. Use -OnHost, or install Docker Desktop."
    }

    if ($Embedder -eq 'openai' -and -not $env:OPENAI_API_KEY -and -not (Test-Path '.env')) {
        # Not fatal: Compose may inject it from elsewhere. Say so now rather than
        # after the query embed call 401s.
        Write-Host "warning: OPENAI_API_KEY not set and no .env present." -ForegroundColor Yellow
    }

    if ($Rerank -eq 'cohere' -and -not $env:COHERE_API_KEY -and -not (Test-Path '.env')) {
        # Same reasoning. The job checks the key before querying, so this only
        # buys an earlier and clearer message.
        Write-Host "warning: COHERE_API_KEY not set and no .env present." -ForegroundColor Yellow
    }

    if ($Rerank -eq 'local') {
        # Worth saying once, loudly: the first local run is slow and needs the
        # network, and inside a fresh container it is slow EVERY time unless the
        # Hugging Face cache is mounted.
        Write-Host "note: --rerank local loads a cross-encoder (~1.1 GB on first use)." -ForegroundColor DarkGray
        Write-Host "      In a container the HF cache is not persisted by default." -ForegroundColor DarkGray
    }

    if ($Rerank -eq 'fake' -and -not $Json) {
        Write-Host "note: 'fake' rerank is a plumbing double, not a quality signal." -ForegroundColor DarkGray
        Write-Host "      Use -Rerank local (or cohere) for an ordering worth reading." -ForegroundColor DarkGray
    }

    if ($Compare) {
        Write-Host ""
        Write-Host "--- baseline: fusion order, no rerank ---" -ForegroundColor Cyan
        $baseline = Invoke-Retrieve -RerankKind 'off'
        if ($baseline -ne 0) { throw "baseline retrieve failed with exit code $baseline" }

        Write-Host ""
        Write-Host "--- reranked: $Rerank ---" -ForegroundColor Cyan
        $code = Invoke-Retrieve -RerankKind $Rerank
    }
    else {
        $code = Invoke-Retrieve -RerankKind $Rerank
    }

    if ($code -ne 0) {
        # Exit 2 is a bad invocation, a collection that cannot serve this mode, or
        # a reranker that cannot be built (missing package, missing key).
        # Retrying will not fix any of them.
        if ($code -eq 2) {
            Write-Host ""
            Write-Host "exit 2 means retrying cannot help. The usual causes:" -ForegroundColor Yellow
            Write-Host "  - collection predates M2 (no 'sparse' vector):  .\scripts\ingest.ps1 -Recreate"
            Write-Host "  - -Rerank local without sentence-transformers:  install it, or use -Rerank fake"
            Write-Host "  - -Rerank cohere without COHERE_API_KEY:        put it in .env, never on the CLI"
        }
        throw "retrieve failed with exit code $code"
    }

    if (-not $Json) {
        Write-Host ""
        if ($Compare) {
            Write-Host "compare the two: a hit whose pre_rerank_rank is far below its rank" -ForegroundColor DarkGray
            Write-Host "is the rerank stage earning its latency. Identical orders mean it" -ForegroundColor DarkGray
            Write-Host "ran and changed nothing - information, not a bug." -ForegroundColor DarkGray
            Write-Host ""
        }
        Write-Host "these are retrieved passages, not an answer." -ForegroundColor DarkGray
        Write-Host "generation with citations is M4." -ForegroundColor DarkGray
    }
}
finally {
    Pop-Location
}
