# Judging-uptime watchdog (Nov 4-8). Run every 5 minutes from Task Scheduler (register_tasks.ps1).
# Restarts the backend after 2 failed /health checks in a row, and restarts ngrok if it died.
param([int]$Port = 8000, [switch]$NoNgrok)
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $repo 'work' }
$logDir = if ($env:SIGHTLINE_LOG_DIR_WIN) { $env:SIGHTLINE_LOG_DIR_WIN } else { Join-Path $work 'logs' }
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "watchdog.log"
if ((Test-Path $log) -and (Get-Item $log).Length -gt 5MB) { Move-Item -Force $log "$log.1" }
$state = Join-Path $logDir "watchdog.state"
function Say($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format s), $m) }

$ok = $false
try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 10; $ok = ($h.status -eq "ok") } catch {}
if ($ok) {
  Set-Content $state 0
} else {
  $fails = 1 + [int](Get-Content $state -ErrorAction SilentlyContinue)
  Set-Content $state $fails
  Say "health failed ($fails in a row)"
  if ($fails -ge 2) {
    Say "restarting backend"
    & "$PSScriptRoot\stop_backend.ps1" | ForEach-Object { Say $_ }
    & "$PSScriptRoot\start_backend.ps1" | ForEach-Object { Say $_ }
    Set-Content $state 0
  }
}
if (-not $NoNgrok -and (Get-Command ngrok -ErrorAction SilentlyContinue) -and -not (Get-Process ngrok -ErrorAction SilentlyContinue)) {
  Say "ngrok not running: starting it"
  & "$PSScriptRoot\start_ngrok.ps1" | ForEach-Object { Say $_ }
}
