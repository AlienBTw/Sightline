# Status of the V-JEPA feature extraction (Sightline phase 03)
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $Repo 'work' }
$LogDir = if ($env:SIGHTLINE_LOG_DIR_WIN) { $env:SIGHTLINE_LOG_DIR_WIN } else { Join-Path $Work 'logs' }
$out = Join-Path $Work 'features\vjepa2_1_vitl_384'
$v = (Get-ChildItem "$out\voxel51" -Filter *.npz -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.tmp.npz' }).Count
$z = (Get-ChildItem "$out\zenodo" -Filter *.npz -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.tmp.npz' }).Count
"voxel51 clips: $v / 658    zenodo images: $z / 12183"
$e = (Get-Content "$out\errors_r*.jsonl" -ErrorAction SilentlyContinue | Measure-Object).Count; "errors: $e"
foreach ($r in 0,1) { "--- rank $r"; Get-Content (Join-Path $LogDir "extract_r$r.log") -Tail 2 -ErrorAction SilentlyContinue }
$size = (Get-ChildItem $out -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB
"features on disk: {0:N1} GiB" -f $size
wsl -d Ubuntu-24.04 -- bash -c "pgrep -af '[e]xtract_features.py' | cut -c1-80; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
