# Sightline: progress of the full vLLM labeling run (started 13:38 Tunis, 2026-10-31).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File training\02-labeling\scripts\check_label_full.ps1
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $Repo 'work' }
$LogDir = if ($env:SIGHTLINE_LOG_DIR_WIN) { $env:SIGHTLINE_LOG_DIR_WIN } else { Join-Path $Work 'logs' }
$status = @(Get-Content (Join-Path $LogDir 'label_full_vllm.status.log'))
$status
$start = [datetime]::ParseExact($status[0].Substring(8,19), 'yyyy-MM-dd HH:mm:ss', $null)
$lines = @(Get-Content (Join-Path $LogDir 'label_full_vllm.out.log'))
$done = ($lines | Select-String '\] +\[\d+/658\]').Count
$fail = ($lines | Select-String 'gave up|failed - clip skipped|FAILED').Count
$errf = "$Repo\training\02-labeling\outputs\weak_labels_v2_vllm.errors.jsonl"
$errn = if (Test-Path $errf) { @(Get-Content $errf).Count } else { 0 }
$files = Get-ChildItem "$Repo\training\cache\glimmer" -Recurse -File -Filter *.json | ? { $_.LastWriteTime -gt $start }
$recent = @($files | ? { $_.LastWriteTime -gt (Get-Date).AddMinutes(-10) }).Count / 10
$left = 658 * 4 - 96 - $files.Count   # 96 = vLLM frames already cached before the run (23 eval clips + smoke clip)
$eta = if ($recent -gt 0) { (Get-Date).AddMinutes($left / $recent).ToString('HH:mm') } else { '?' }
"clips done {0}/658; new frames {1}; rate last 10 min {2:N1} calls/min; ~{3} calls left; ETA {4} Tunis; error lines {5}; errors.jsonl rows {6}" -f $done, $files.Count, $recent, $left, $eta, $fail, $errn
"runner PID 7400 alive: $([bool](Get-Process -Id 7400 -EA SilentlyContinue)); vLLM up: $(try { [bool](Invoke-RestMethod http://localhost:30001/v1/models -TimeoutSec 5) } catch { $false })"
$lines | Select -Last 2
