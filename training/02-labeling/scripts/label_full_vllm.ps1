# Sightline: full v2/none Glimmer labeling on vLLM (c=32). Detached; logs under work\logs\label_full_vllm.*.log
$ErrorActionPreference = 'Continue'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $Repo 'work' }
$LogDir = if ($env:SIGHTLINE_LOG_DIR_WIN) { $env:SIGHTLINE_LOG_DIR_WIN } else { Join-Path $Work 'logs' }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$st = Join-Path $LogDir 'label_full_vllm.status.log'
$outLog = Join-Path $LogDir 'label_full_vllm.out.log'
$errLog = Join-Path $LogDir 'label_full_vllm.err.log'
"started $(Get-Date -f 'yyyy-MM-dd HH:mm:ss') Tunis pid $PID" | Set-Content $st
python -X utf8 training/02-labeling/label_glimmer.py --backend vllm --sources voxel51_safe_unsafe --prompt training/02-labeling/prompts/factors_v2.md --reasoning-effort none --concurrency 32 --out training/02-labeling/outputs/weak_labels_v2_vllm.jsonl 1> $outLog 2> $errLog
"finished $(Get-Date -f 'yyyy-MM-dd HH:mm:ss') Tunis EXIT $LASTEXITCODE" | Add-Content $st
