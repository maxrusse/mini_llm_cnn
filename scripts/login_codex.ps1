param(
    [string]$WorkspaceRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = Split-Path -Parent $PSScriptRoot
}

$codexCmd = Get-Command codex -ErrorAction SilentlyContinue
if ($null -eq $codexCmd) {
    throw "codex CLI not found in PATH."
}

$codexHome = Join-Path $WorkspaceRoot ".mini_loop\codex_home"
if (-not (Test-Path $codexHome)) {
    New-Item -ItemType Directory -Path $codexHome -Force | Out-Null
}

$prevCodexHome = $env:CODEX_HOME
$env:CODEX_HOME = $codexHome
try {
    Write-Host ("Using CODEX_HOME=" + $codexHome)
    & codex login --device-auth
    & codex login status
} finally {
    if ([string]::IsNullOrWhiteSpace($prevCodexHome)) {
        Remove-Item Env:CODEX_HOME -ErrorAction SilentlyContinue
    } else {
        $env:CODEX_HOME = $prevCodexHome
    }
}
