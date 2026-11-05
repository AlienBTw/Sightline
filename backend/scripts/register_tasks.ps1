# One-time setup for judging (run in an elevated PowerShell if your policy requires it).
# Registers a single "Sightline Watchdog" task for the current user:
#   - at logon
#   - every 5 minutes, repeating for 3650 days
# The task runs hidden and ignores a new instance if one is already running.
# No work, log, or repo paths are passed; the script reads SIGHTLINE_* env vars at runtime.
# Older "Sightline Backend" and "Sightline ngrok" logon tasks are removed so they cannot
# start a second copy. Remove later with:
#   Unregister-ScheduledTask -TaskName "Sightline*" -Confirm:$false
$ps = "powershell.exe"
$base = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File"
$here = $PSScriptRoot
$user = $env:USERNAME
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
# A long RepetitionDuration keeps the 5-minute repeat from stopping after the default window.
$every5 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
  -RepetitionInterval (New-TimeSpan -Minutes 5) `
  -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$action = New-ScheduledTaskAction -Execute $ps -Argument "$base `"$here\watchdog.ps1`""
Register-ScheduledTask -Force -TaskName "Sightline Watchdog" `
  -Trigger @($logon, $every5) -Settings $settings -Action $action -Principal $principal `
  -Description "Ensure the Sightline camera sim, backend and ngrok are up. Env vars are read at runtime."
foreach ($old in @("Sightline Backend", "Sightline ngrok")) {
  if (Get-ScheduledTask -TaskName $old -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $old -Confirm:$false
  }
}
Write-Output "Registered: Sightline Watchdog (at logon and every 5 minutes). Separate Backend and ngrok logon tasks removed if they existed."
