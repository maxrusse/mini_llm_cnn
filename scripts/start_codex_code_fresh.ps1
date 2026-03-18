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
    (Join-Path $repoRoot ".mini_loop\codex_home_code"),
    (Join-Path $repoRoot "logs_code\.gitkeep"),
    (Join-Path $repoRoot "generated_configs_code\.gitkeep"),
    (Join-Path $repoRoot "runs_code\.gitkeep"),
    (Join-Path $repoRoot "downloads_code\.gitkeep")
)

foreach ($path in @(
    ".mini_loop\codex_session_code.json",
    ".mini_loop\codex_thread_id_code.txt",
    ".mini_loop\launcher_session_code.json",
    ".mini_loop\STOP_CODEX_LOOP_CODE",
    "results_code.tsv",
    "experiment_summary_code.tsv"
)) {
    if (Test-Path $path) {
        Remove-Item $path -Force -Recurse
    }
}

foreach ($dir in @("logs_code", "generated_configs_code", "runs_code", "downloads_code")) {
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
    SearchSpace = "code"
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
