# Status of a detached APK build: is it running, last log lines, and the output APK.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build-status.ps1
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $repo 'work' }
. "$PSScriptRoot\..\..\backend\scripts\_logdir.ps1"
$logDir = Get-SightlineLogDir
$log = Join-Path $logDir 'mobile_build.log'
$pidFile = Join-Path $logDir 'mobile_build.pid'
if (Test-Path $pidFile) {
  $p = Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue
  if ($p) { Write-Output "RUNNING (pid $($p.Id), started $($p.StartTime))" } else { Write-Output "NOT RUNNING" }
}
if (Test-Path $log) { Get-Content $log -Tail 8 }
$errLog = Join-Path $logDir 'mobile_build.err.log'
if (Test-Path $errLog) { Get-Content $errLog -Tail 4 }
$apk = Get-Item (Join-Path $work 'sightline_release\sightline.apk') -ErrorAction SilentlyContinue
if ($apk) { Write-Output ("APK: {0} {1:N1} MB, {2}" -f $apk.FullName, ($apk.Length / 1MB), $apk.LastWriteTime) }
