# Sightline: full v2/none Glimmer labeling on vLLM (c=32). Detached; logs F:\models\logs\label_full_vllm.*.log
$ErrorActionPreference = 'Continue'
Set-Location 'C:\projects\Meta Global AI Developer Hackathon\Sightline'
$st = 'F:\models\logs\label_full_vllm.status.log'
"started $(Get-Date -f 'yyyy-MM-dd HH:mm:ss') Tunis pid $PID" | Set-Content $st
python -X utf8 training/02-labeling/label_glimmer.py --backend vllm --sources voxel51_safe_unsafe --prompt training/02-labeling/prompts/factors_v2.md --reasoning-effort none --concurrency 32 --out training/02-labeling/outputs/weak_labels_v2_vllm.jsonl 1> F:\models\logs\label_full_vllm.out.log 2> F:\models\logs\label_full_vllm.err.log
"finished $(Get-Date -f 'yyyy-MM-dd HH:mm:ss') Tunis EXIT $LASTEXITCODE" | Add-Content $st