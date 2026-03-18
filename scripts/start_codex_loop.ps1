param(
    [string]$Tier = "medium",
    [double]$Hours = 8.0,
    [ValidateSet("open", "limited", "code", "aggressive")]
    [string]$SearchSpace = "open",
    [string]$SeedParentExperimentId = "",
    [switch]$StartFromScratch,
    [switch]$StartInNewWindow,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$loopConfig = if ($SearchSpace -eq "code") {
    Join-Path $repoRoot "config\codex_loop_code.json"
} elseif ($SearchSpace -eq "aggressive") {
    Join-Path $repoRoot "config\codex_loop_aggressive.json"
} else {
    Join-Path $repoRoot "config\codex_loop.json"
}
if (-not (Test-Path $loopConfig)) {
    throw "Missing loop config: $loopConfig"
}

$controllerCfg = Get-Content $loopConfig -Raw | ConvertFrom-Json
$pythonExe = [string]$controllerCfg.controller_python_exe
if (-not [System.IO.Path]::IsPathRooted($pythonExe)) {
    $pythonExe = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $pythonExe))
}
if (-not (Test-Path $pythonExe)) {
    throw "Configured controller_python_exe does not exist: $pythonExe"
}

$runnerPath = Join-Path $repoRoot "scripts\codex_loop.py"
if (-not (Test-Path $runnerPath)) {
    throw "Missing codex loop runner: $runnerPath"
}

$searchSpaceDoc = if ($SearchSpace -eq "limited") {
    Join-Path $repoRoot "search_space_limited.md"
} elseif ($SearchSpace -eq "code") {
    Join-Path $repoRoot "search_space_code.md"
} elseif ($SearchSpace -eq "aggressive") {
    Join-Path $repoRoot "search_space_aggressive.md"
} else {
    Join-Path $repoRoot "search_space_open.md"
}
if (-not (Test-Path $searchSpaceDoc)) {
    throw "Missing search-space doc: $searchSpaceDoc"
}

$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$metaDir = Join-Path $repoRoot ".mini_loop"
New-Item -ItemType Directory -Force $metaDir | Out-Null
$stamp = "{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff"), ([guid]::NewGuid().ToString("N").Substring(0, 6))
$sessionLog = Join-Path $logDir ("codex_launcher_{0}.log" -f $stamp)
$sessionMetaName = if ($SearchSpace -eq "open") {
    "launcher_session.json"
} else {
    "launcher_session_{0}.json" -f $SearchSpace
}
$sessionMeta = Join-Path $metaDir $sessionMetaName

$argList = @(
    $runnerPath,
    "--config-path", $loopConfig,
    "--tier", $Tier,
    "--hours", $Hours.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--search-space", $SearchSpace
)
if ($DryRun) {
    $argList += "--dry-run"
}
if (-not [string]::IsNullOrWhiteSpace($SeedParentExperimentId)) {
    $argList += @("--seed-parent-experiment-id", $SeedParentExperimentId)
}
if ($StartFromScratch) {
    $argList += "--start-from-scratch"
}

$cmdPreview = @("&", $pythonExe) + ($argList | ForEach-Object {
    if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
})
$previewText = $cmdPreview -join " "

"[$((Get-Date).ToString('o'))] LAUNCH $previewText" | Out-File -FilePath $sessionLog -Append -Encoding utf8
Write-Host "Repo: $repoRoot"
Write-Host "Python: $pythonExe"
Write-Host "Log: $sessionLog"
Write-Host "Search space: $SearchSpace ($searchSpaceDoc)"
Write-Host "Command: $previewText"

@{
    started_utc = (Get-Date).ToUniversalTime().ToString("o")
    tier = $Tier
    hours = $Hours
    search_space = $SearchSpace
    search_space_doc = $searchSpaceDoc
    launcher_log = $sessionLog
    dry_run = [bool]$DryRun
} | ConvertTo-Json | Set-Content -Path $sessionMeta -Encoding UTF8

if ($StartInNewWindow) {
    $pwsh = (Get-Command pwsh -ErrorAction Stop).Source
    $childArgs = $argList | ForEach-Object {
        $txt = [string]$_
        if ($txt -match '[\s"]') {
            '"' + ($txt -replace '"', '\"') + '"'
        } else {
            $txt
        }
    }
    $childCommand = @(
        "Set-Location '$repoRoot'",
        "& '$pythonExe' " + ($childArgs -join " ")
    ) -join "; "
    $proc = Start-Process -FilePath $pwsh -ArgumentList @("-NoExit", "-Command", $childCommand) -WorkingDirectory $repoRoot -PassThru
    "[$((Get-Date).ToString('o'))] PID $($proc.Id)" | Out-File -FilePath $sessionLog -Append -Encoding utf8
    Write-Host "Started new window with PID $($proc.Id)"
    exit 0
}

& $pythonExe @argList 2>&1 | Tee-Object -FilePath $sessionLog -Append
exit $LASTEXITCODE
