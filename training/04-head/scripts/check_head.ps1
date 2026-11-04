# Status of Sightline 04-head runs
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $Repo 'work' }
$LogDir = if ($env:SIGHTLINE_LOG_DIR_WIN) { $env:SIGHTLINE_LOG_DIR_WIN } else { Join-Path $Work 'logs' }
Get-Content (Join-Path $LogDir 'head_sweep.log'), (Join-Path $LogDir 'head_cv.log') -ErrorAction SilentlyContinue
Get-ChildItem (Join-Path $LogDir 'head_h*.log') | Sort-Object LastWriteTime | ForEach-Object { "{0,-28} {1}" -f $_.BaseName, ((Get-Content $_.FullName -Tail 1) -replace '^(.{0,90}).*','$1') }
Get-ChildItem (Join-Path $Repo 'training\04-head\results') -Filter *.json | ForEach-Object Name
wsl -d Ubuntu-24.04 -- bash -c 'pgrep -af "[t]rain_head.py" | cut -c1-110'
