param(
    [string]$Tier = "medium",
    [double]$Hours = 8.0,
    [ValidateSet("open", "limited")]
    [string]$SearchSpace = "open",
    [switch]$StartInNewWindow,
    [switch]$DryRun
)

$scriptPath = Join-Path $PSScriptRoot "start_codex_loop.ps1"
if (-not (Test-Path $scriptPath)) {
    throw "Missing script: $scriptPath"
}

& $scriptPath -Tier $Tier -Hours $Hours -SearchSpace $SearchSpace -StartInNewWindow:$StartInNewWindow -DryRun:$DryRun
exit $LASTEXITCODE
