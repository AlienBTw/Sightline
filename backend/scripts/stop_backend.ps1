# Stop the Sightline backend (graceful, then forced).
param([string]$Distro = "Ubuntu-24.04")
. "$PSScriptRoot\_logdir.ps1"
$logDir = Get-SightlineLogDir
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$wslRepo = "/mnt/" + $repo.Substring(0,1).ToLower() + ($repo.Substring(2) -replace '\\','/')
wsl.exe -d $Distro -- bash "$wslRepo/backend/scripts/stop_backend.sh"
if ($LASTEXITCODE -eq 0) {
  $pidFile = Join-Path $logDir 'sightline_backend_wsl.pid'
  if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
}
