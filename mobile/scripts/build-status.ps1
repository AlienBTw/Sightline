# Status of a detached APK build: is it running, last log lines, and the output APK.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build-status.ps1
$log = "F:\models\logs\mobile_build.log"
$pidFile = "F:\models\logs\mobile_build.pid"
if (Test-Path $pidFile) {
  $p = Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue
  if ($p) { Write-Output "RUNNING (pid $($p.Id), started $($p.StartTime))" } else { Write-Output "NOT RUNNING" }
}
if (Test-Path $log) { Get-Content $log -Tail 8 }
if (Test-Path "F:\models\logs\mobile_build.err.log") { Get-Content "F:\models\logs\mobile_build.err.log" -Tail 4 }
$apk = Get-Item "F:\sightline_release\sightline.apk" -ErrorAction SilentlyContinue
if ($apk) { Write-Output ("APK: {0} {1:N1} MB, {2}" -f $apk.FullName, ($apk.Length / 1MB), $apk.LastWriteTime) }