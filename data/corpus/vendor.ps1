param(
    [Parameter(Mandatory = $true)]
    [string]$UpstreamCheckout,

    [string]$Destination = (Join-Path $PSScriptRoot "qdrant")
)

$ErrorActionPreference = "Stop"
$expectedCommit = "cc9f98286dd98eca3c5bc57110b50887ca0da446"
$actualCommit = (git -C $UpstreamCheckout rev-parse HEAD).Trim()

if ($actualCommit -ne $expectedCommit) {
    throw "Expected landing_page commit $expectedCommit, found $actualCommit"
}

$source = Join-Path $UpstreamCheckout "qdrant-landing/content/documentation"
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Qdrant documentation directory not found: $source"
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
Get-ChildItem -LiteralPath $source -Recurse -File -Filter "*.md" | ForEach-Object {
    $relative = [System.IO.Path]::GetRelativePath($source, $_.FullName)
    $target = Join-Path $Destination $relative
    New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $target -Force
}

$documents = Get-ChildItem -LiteralPath $Destination -Recurse -File -Filter "*.md"
Write-Output "Vendored $($documents.Count) Markdown documents from $actualCommit."
