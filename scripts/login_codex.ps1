param(
    [string]$WorkspaceRoot = "",
    [ValidateSet("open", "limited", "code", "aggressive", "all")]
    [string]$SearchSpace = "all"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = Split-Path -Parent $PSScriptRoot
}

$codexCmd = Get-Command codex -ErrorAction SilentlyContinue
if ($null -eq $codexCmd) {
    throw "codex CLI not found in PATH."
}

$codexHomes = @()
switch ($SearchSpace) {
    "code" {
        $codexHomes += (Join-Path $WorkspaceRoot ".mini_loop\codex_home_code")
    }
    "aggressive" {
        $codexHomes += (Join-Path $WorkspaceRoot ".mini_loop\codex_home_aggressive")
    }
    "open" {
        $codexHomes += (Join-Path $WorkspaceRoot ".mini_loop\codex_home")
    }
    "limited" {
        $codexHomes += (Join-Path $WorkspaceRoot ".mini_loop\codex_home")
    }
    default {
        $codexHomes += (Join-Path $WorkspaceRoot ".mini_loop\codex_home")
        $codexHomes += (Join-Path $WorkspaceRoot ".mini_loop\codex_home_code")
        $codexHomes += (Join-Path $WorkspaceRoot ".mini_loop\codex_home_aggressive")
    }
}
$codexHomes = $codexHomes | Select-Object -Unique

foreach ($codexHome in $codexHomes) {
    if (-not (Test-Path $codexHome)) {
        New-Item -ItemType Directory -Path $codexHome -Force | Out-Null
    }
}

$prevCodexHome = $env:CODEX_HOME
try {
    foreach ($codexHome in $codexHomes) {
        $env:CODEX_HOME = $codexHome
        Write-Host ("Using CODEX_HOME=" + $codexHome)
        & codex login --device-auth
        & codex login status
    }
} finally {
    if ([string]::IsNullOrWhiteSpace($prevCodexHome)) {
        Remove-Item Env:CODEX_HOME -ErrorAction SilentlyContinue
    } else {
        $env:CODEX_HOME = $prevCodexHome
    }
}
