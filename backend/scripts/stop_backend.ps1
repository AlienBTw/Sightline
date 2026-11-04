# Stop the Sightline backend (graceful, then forced).
param([string]$Distro = "Ubuntu-24.04")
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$wslRepo = "/mnt/" + $repo.Substring(0,1).ToLower() + ($repo.Substring(2) -replace '\\','/')
wsl.exe -d $Distro -- bash "$wslRepo/backend/scripts/stop_backend.sh"
