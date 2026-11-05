# Judging-uptime watchdog. Ensures the camera sim, the backend and ngrok are up.
# Run at logon and every 5 minutes (register_tasks.ps1).
# Starts only what is down. A healthy service is left alone, and a second copy is not started.
#   powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\watchdog.ps1 [-Report]
# -Report / -DryRun prints one status line per service (healthy | would start) and starts nothing.
param(
  [int]$Port = 8000,
  [string]$Distro = "Ubuntu-24.04",
  [switch]$NoNgrok,
  [switch]$NoSim,
  [Alias('Report')]
  [switch]$DryRun
)
. "$PSScriptRoot\_logdir.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$wslRepo = "/mnt/" + $repo.Substring(0,1).ToLower() + ($repo.Substring(2) -replace '\\','/')
$logDir = Get-SightlineLogDir
$log = Join-Path $logDir 'watchdog.log'
if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).Length -gt 5MB) {
  Move-Item -Force -LiteralPath $log "$log.1"
}
$state = Join-Path $logDir 'watchdog.state'
function Say($m) { Add-Content -LiteralPath $log -Value ("{0} {1}" -f (Get-Date -Format s), $m) }

# Run one bash line inside WSL. Used for pid checks only; it does not start services.
function Invoke-WslBashLine([string]$Line) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = 'wsl.exe'
  $psi.Arguments = "-d $Distro -- bash -s"
  $psi.RedirectStandardInput = $true
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  $proc = $null
  try {
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.StandardInput.Write(($Line -replace "`r", '') + "`n")
    $proc.StandardInput.Close()
    if (-not $proc.WaitForExit(20000)) {
      try { $proc.Kill() } catch {}
      return $false
    }
    return ($proc.ExitCode -eq 0)
  } catch {
    if ($proc -and -not $proc.HasExited) { try { $proc.Kill() } catch {} }
    return $false
  } finally {
    if ($proc) { $proc.Dispose() }
  }
}

$repoQ = $wslRepo -replace "'", "'\''"
# Same defaults as tools/camera_sim/start_sim.sh: $CAMSIM_LOG or <work>/logs/camsim.
$simLine = @'
REPO='__REPO__'; WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"; LOG="${CAMSIM_LOG:-$WORK/logs/camsim}"; pid=$(cat "$LOG/mediamtx.pid" 2>/dev/null || true); kill -0 "$pid" 2>/dev/null || exit 1; shopt -s nullglob; pubs=("$LOG"/pub_*.pid); [[ ${#pubs[@]} -gt 0 ]] || exit 1; for f in "$LOG"/*.pid; do pid=$(cat "$f" 2>/dev/null || true); kill -0 "$pid" 2>/dev/null || exit 1; done; exit 0
'@.Replace('__REPO__', $repoQ).Trim()
# Same defaults as backend/scripts/run_backend.sh for the in-WSL pid file.
$backendLine = @'
REPO='__REPO__'; WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"; LOG="${SIGHTLINE_LOG_DIR:-$WORK/logs}"; pid=$(cat "$LOG/sightline_backend.pid" 2>/dev/null || true); [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
'@.Replace('__REPO__', $repoQ).Trim()

function Test-BackendProcessAlive {
  $win = $false
  $pidFile = Join-Path $logDir 'sightline_backend_wsl.pid'
  if (Test-Path -LiteralPath $pidFile) {
    $raw = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($raw -match '^\s*(\d+)\s*$') {
      $proc = Get-Process -Id ([int]$Matches[1]) -ErrorAction SilentlyContinue
      if ($proc -and $proc.ProcessName -match '^wsl') { $win = $true }
    }
  }
  $linux = Invoke-WslBashLine $backendLine
  return ($win -or $linux)
}

$report = @()

# 1) Camera sim. start_sim.sh skips any pid that is already alive.
if (-not $NoSim) {
  if (Invoke-WslBashLine $simLine) {
    $report += 'sim: healthy'
  } else {
    $report += 'sim: would start'
    if ($DryRun) {
      Say 'camera sim not healthy: would start start_sim.sh'
    } else {
      Say 'camera sim not healthy: starting start_sim.sh'
      Start-Process -FilePath 'wsl.exe' -WindowStyle Hidden `
        -ArgumentList "-d $Distro -- bash `"$wslRepo/tools/camera_sim/start_sim.sh`""
    }
  }
}

# 2) Backend. Two failed /health checks before a restart, unless no process exists at all.
$healthOk = $false
try {
  $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 10
  $healthOk = ($h.status -eq 'ok')
} catch {}
if ($healthOk) {
  $report += 'backend: healthy'
  if (-not $DryRun) { Set-Content -LiteralPath $state -Value '0' }
} else {
  $alive = Test-BackendProcessAlive
  $prev = 0
  $rawState = Get-Content -LiteralPath $state -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($rawState -match '^\s*(\d+)') { $prev = [int]$Matches[1] }
  $fails = $prev + 1
  if (-not $alive) {
    $report += 'backend: would start'
    if ($DryRun) {
      Say 'health failed and no backend process is alive: would start'
    } else {
      Say 'health failed and no backend process is alive: starting'
      & "$PSScriptRoot\start_backend.ps1" -Distro $Distro -Port $Port | ForEach-Object { Say $_ }
      Set-Content -LiteralPath $state -Value '0'
    }
  } elseif ($fails -ge 2) {
    $report += 'backend: would start'
    if ($DryRun) {
      Say "health failed ($fails in a row): would restart backend"
    } else {
      Say "health failed ($fails in a row): restarting backend"
      & "$PSScriptRoot\stop_backend.ps1" -Distro $Distro | ForEach-Object { Say $_ }
      & "$PSScriptRoot\start_backend.ps1" -Distro $Distro -Port $Port | ForEach-Object { Say $_ }
      Set-Content -LiteralPath $state -Value '0'
    }
  } else {
    # Process is alive and this is the first miss: leave it. The line stays "healthy"
    # because this run would not start or restart it. The miss is still logged.
    $report += 'backend: healthy'
    Say "health failed ($fails in a row); backend process still alive"
    if (-not $DryRun) { Set-Content -LiteralPath $state -Value "$fails" }
  }
}

# 3) ngrok, as before: start only when the command exists and the process does not.
if (-not $NoNgrok) {
  $ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue
  $ngrokProc = Get-Process ngrok -ErrorAction SilentlyContinue
  if ($ngrokProc) {
    $report += 'ngrok: healthy'
  } elseif ($ngrokCmd) {
    $report += 'ngrok: would start'
    if ($DryRun) {
      Say 'ngrok not running: would start'
    } else {
      Say 'ngrok not running: starting it'
      & "$PSScriptRoot\start_ngrok.ps1" -Port $Port | ForEach-Object { Say $_ }
    }
  } else {
    $report += 'ngrok: healthy'
    Say 'ngrok not on PATH: not starting'
  }
}

if ($DryRun) {
  foreach ($line in $report) { Say $line; Write-Output $line }
}
