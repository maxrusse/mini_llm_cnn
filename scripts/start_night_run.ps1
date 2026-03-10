param(
    [string]$Tier = "medium",
    [double]$Hours = 8.0,
    [int]$SleepSeconds = 5,
    [ValidateSet("open", "limited")]
    [string]$SearchSpace = "open",
    [switch]$StartInNewWindow,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repoRoot "config.json"
if (-not (Test-Path $configPath)) {
    throw "Missing config.json at $configPath"
}

$config = Get-Content $configPath -Raw | ConvertFrom-Json
$pythonExe = [string]$config.benchmark_python_exe
if (-not [System.IO.Path]::IsPathRooted($pythonExe)) {
    $pythonExe = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $pythonExe))
}
if (-not (Test-Path $pythonExe)) {
    throw "Configured benchmark_python_exe does not exist: $pythonExe"
}

$runLoopPath = Join-Path $repoRoot "run_loop.py"
if (-not (Test-Path $runLoopPath)) {
    throw "Missing run_loop.py at $runLoopPath"
}

$searchSpaceDoc = if ($SearchSpace -eq "limited") {
    Join-Path $repoRoot "search_space_limited.md"
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
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionLog = Join-Path $logDir ("night_run_launcher_{0}.log" -f $stamp)
$sessionMeta = Join-Path $metaDir "launcher_session.json"

$argList = @(
    $runLoopPath,
    "night-run",
    "--tier", $Tier,
    "--hours", $Hours.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--sleep-seconds", "$SleepSeconds"
)
if ($DryRun) {
    $argList += "--dry-run"
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
    sleep_seconds = $SleepSeconds
    search_space = $SearchSpace
    search_space_doc = $searchSpaceDoc
    launcher_log = $sessionLog
    dry_run = [bool]$DryRun
} | ConvertTo-Json | Set-Content -Path $sessionMeta -Encoding UTF8

if ($StartInNewWindow) {
    $pwsh = (Get-Command pwsh -ErrorAction Stop).Source
    $childCommand = @(
        "Set-Location '$repoRoot'",
        "& '$pythonExe' '$runLoopPath' night-run --tier $Tier --hours $($Hours.ToString([System.Globalization.CultureInfo]::InvariantCulture)) --sleep-seconds $SleepSeconds" + $(if ($DryRun) { " --dry-run" } else { "" })
    ) -join "; "
    $proc = Start-Process -FilePath $pwsh -ArgumentList @("-NoExit", "-Command", $childCommand) -WorkingDirectory $repoRoot -PassThru
    "[$((Get-Date).ToString('o'))] PID $($proc.Id)" | Out-File -FilePath $sessionLog -Append -Encoding utf8
    Write-Host "Started new window with PID $($proc.Id)"
    exit 0
}

& $pythonExe @argList 2>&1 | Tee-Object -FilePath $sessionLog -Append
exit $LASTEXITCODE
