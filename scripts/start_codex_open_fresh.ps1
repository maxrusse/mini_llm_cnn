param(
    [string]$Tier = "medium",
    [double]$Hours = 8.0,
    [switch]$StartInNewWindow,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$keepPaths = @(
    (Join-Path $repoRoot ".mini_loop\codex_home"),
    (Join-Path $repoRoot "logs\.gitkeep"),
    (Join-Path $repoRoot "generated_configs\.gitkeep"),
    (Join-Path $repoRoot "runs\.gitkeep"),
    (Join-Path $repoRoot "downloads\.gitkeep")
)

foreach ($path in @(
    ".mini_loop\codex_session.json",
    ".mini_loop\codex_thread_id.txt",
    ".mini_loop\launcher_session.json",
    ".mini_loop\state.json",
    ".mini_loop\STOP_CODEX_LOOP",
    "results.tsv",
    "experiment_summary.tsv"
)) {
    if (Test-Path $path) {
        Remove-Item $path -Force -Recurse
    }
}

foreach ($dir in @("logs", "generated_configs", "runs", "downloads")) {
    if (-not (Test-Path $dir)) {
        continue
    }
    Get-ChildItem -Force $dir | ForEach-Object {
        if ($_.FullName -notin $keepPaths) {
            Remove-Item $_.FullName -Force -Recurse
        }
    }
}

$launchParams = @{
    Tier = $Tier
    Hours = $Hours
    SearchSpace = "open"
}
if ($StartInNewWindow) {
    $launchParams.StartInNewWindow = $true
}
if ($DryRun) {
    $launchParams.DryRun = $true
}

& (Join-Path $PSScriptRoot "start_codex_loop.ps1") @launchParams
exit $LASTEXITCODE
