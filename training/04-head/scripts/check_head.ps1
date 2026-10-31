# Status of Sightline 04-head runs
Get-Content F:\models\logs\head_sweep.log, F:\models\logs\head_cv.log -ErrorAction SilentlyContinue
Get-ChildItem F:\models\logs\head_h*.log | Sort LastWriteTime | % { "{0,-28} {1}" -f $_.BaseName, ((Get-Content $_.FullName -Tail 1) -replace '^(.{0,90}).*','$1') }
Get-ChildItem "C:\projects\Meta Global AI Developer Hackathon\Sightline\training\04-head\results" -Filter *.json | % Name
wsl -d Ubuntu-24.04 -- bash -c 'pgrep -af "[t]rain_head.py" | cut -c1-110'
