# Windows wrapper: start the CCTV simulator inside WSL (detached).  Args are passed through:
#   powershell -ExecutionPolicy Bypass -File tools\camera_sim\start_sim.ps1            # everything
#   powershell -ExecutionPolicy Bypass -File tools\camera_sim\start_sim.ps1 pub 101
param([string]$Distro = "Ubuntu-24.04", [Parameter(ValueFromRemainingArguments = $true)] $Rest)
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$wslRepo = "/mnt/" + $repo.Substring(0,1).ToLower() + ($repo.Substring(2) -replace '\\','/')
wsl.exe -d $Distro -- bash "$wslRepo/tools/camera_sim/start_sim.sh" @Rest
# MediaMTX listens inside WSL; Windows reaches it on localhost:8554 (WSL localhost forwarding).
# For a phone/another PC on the LAN, forward the port (admin PowerShell):
#   netsh interface portproxy add v4tov4 listenport=8554 connectport=8554 connectaddress=$(wsl hostname -I)
