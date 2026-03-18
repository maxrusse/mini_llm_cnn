param(
    [string]$Tier = "medium",
    [double]$Hours = 8.0,
    [string]$SeedParentExperimentId = "",
    [switch]$StartFromScratch,
    [switch]$StartInNewWindow,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$keepPaths = @(
    (Join-Path $repoRoot ".mini_loop\codex_home_aggressive")
)

foreach ($path in @(
    ".mini_loop\codex_session_aggressive.json",
    ".mini_loop\codex_thread_id_aggressive.txt",
    ".mini_loop\launcher_session_aggressive.json",
    ".mini_loop\state_aggressive.json",
    ".mini_loop\STOP_CODEX_LOOP_AGGRESSIVE",
    "results_aggressive.tsv",
    "experiment_summary_aggressive.tsv"
)) {
    if (Test-Path $path) {
        Remove-Item $path -Force -Recurse
    }
}

foreach ($dir in @("logs_aggressive", "generated_configs_aggressive", "runs_aggressive", "downloads_aggressive")) {
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
    SearchSpace = "aggressive"
}
if (-not [string]::IsNullOrWhiteSpace($SeedParentExperimentId)) {
    $launchParams.SeedParentExperimentId = $SeedParentExperimentId
}
if ($StartFromScratch) {
    $launchParams.StartFromScratch = $true
}
if ($StartInNewWindow) {
    $launchParams.StartInNewWindow = $true
}
if ($DryRun) {
    $launchParams.DryRun = $true
}

& (Join-Path $PSScriptRoot "start_codex_loop.ps1") @launchParams
exit $LASTEXITCODE
