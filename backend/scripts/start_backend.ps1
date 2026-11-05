# Start the Sightline backend detached (hidden wsl.exe keeps the WSL VM alive) and wait for /health.
#   powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\start_backend.ps1
# If SIGHTLINE_BACKEND_LAUNCHER is a WSL path to a bash script, that script runs instead of
# backend/scripts/run_backend.sh. -Launcher overrides the env var. /health is checked first;
# an already-up backend is left running.
param(
  [string]$Distro = "Ubuntu-24.04",
  [int]$Port = 8000,
  [int]$WaitSeconds = 90,
  [string]$Launcher = ""
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_logdir.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$wslRepo = "/mnt/" + $repo.Substring(0,1).ToLower() + ($repo.Substring(2) -replace '\\','/')
$logDir = Get-SightlineLogDir
try {
  $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
  Write-Output "already up: $($h.status)"
  exit 0
} catch {}
if ($PSBoundParameters.ContainsKey('Launcher')) {
  $launch = $Launcher
} elseif ($env:SIGHTLINE_BACKEND_LAUNCHER) {
  $launch = $env:SIGHTLINE_BACKEND_LAUNCHER
} else {
  $launch = "$wslRepo/backend/scripts/run_backend.sh"
}
if (-not $launch) { throw "SIGHTLINE_BACKEND_LAUNCHER / -Launcher is empty" }
$p = Start-Process -FilePath "wsl.exe" -WindowStyle Hidden -PassThru `
     -ArgumentList "-d $Distro -- bash `"$launch`""
Set-Content -Path (Join-Path $logDir 'sightline_backend_wsl.pid') -Value $p.Id
$t0 = Get-Date
while (((Get-Date) - $t0).TotalSeconds -lt $WaitSeconds) {
  Start-Sleep -Seconds 2
  try {
    $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
    Write-Output "backend up: $($h.status) (wsl.exe pid $($p.Id))"
    exit 0
  } catch {}
}
Write-Output "backend did not answer /health within $WaitSeconds s; see $logDir\backend_stdout.log and backend.log"
exit 1
