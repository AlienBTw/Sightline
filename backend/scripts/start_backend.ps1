# Start the Sightline backend detached (hidden wsl.exe keeps the WSL VM alive) and wait for /health.
#   powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\start_backend.ps1
param([string]$Distro = "Ubuntu-24.04", [int]$Port = 8000, [int]$WaitSeconds = 90)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$wslRepo = "/mnt/" + $repo.Substring(0,1).ToLower() + ($repo.Substring(2) -replace '\\','/')
$work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $repo 'work' }
$logDir = if ($env:SIGHTLINE_LOG_DIR_WIN) { $env:SIGHTLINE_LOG_DIR_WIN } else { Join-Path $work 'logs' }
New-Item -ItemType Directory -Force $logDir | Out-Null
try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3; Write-Output "already up: $($h.status)"; exit 0 } catch {}
$p = Start-Process -FilePath "wsl.exe" -WindowStyle Hidden -PassThru `
     -ArgumentList "-d $Distro -- bash `"$wslRepo/backend/scripts/run_backend.sh`""
Set-Content -Path "$logDir\sightline_backend_wsl.pid" -Value $p.Id
$t0 = Get-Date
while (((Get-Date) - $t0).TotalSeconds -lt $WaitSeconds) {
  Start-Sleep -Seconds 2
  try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3; Write-Output "backend up: $($h.status) (wsl.exe pid $($p.Id))"; exit 0 } catch {}
}
Write-Output "backend did not answer /health within $WaitSeconds s; see $logDir\backend_stdout.log and backend.log"
exit 1
