param([string]$Distro = "Ubuntu-24.04", [Parameter(ValueFromRemainingArguments = $true)] $Rest)
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$wslRepo = "/mnt/" + $repo.Substring(0,1).ToLower() + ($repo.Substring(2) -replace '\\','/')
wsl.exe -d $Distro -- bash "$wslRepo/tools/camera_sim/stop_sim.sh" @Rest
