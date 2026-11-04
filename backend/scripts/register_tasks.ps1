# One-time setup for judging (run in an elevated PowerShell if your policy requires it):
#   - "Sightline Backend": start the backend at logon
#   - "Sightline ngrok":   start the ngrok tunnel at logon
#   - "Sightline Watchdog": check /health every 5 minutes, restart backend / ngrok if needed
# Remove later with: Unregister-ScheduledTask -TaskName "Sightline*" -Confirm:$false
$ps = "powershell.exe"
$base = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File"
$here = $PSScriptRoot
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
Register-ScheduledTask -Force -TaskName "Sightline Backend" -Trigger $logon -Settings $settings `
  -Action (New-ScheduledTaskAction -Execute $ps -Argument "$base `"$here\start_backend.ps1`"")
Register-ScheduledTask -Force -TaskName "Sightline ngrok" -Trigger $logon -Settings $settings `
  -Action (New-ScheduledTaskAction -Execute $ps -Argument "$base `"$here\start_ngrok.ps1`"")
$every5 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 5)
Register-ScheduledTask -Force -TaskName "Sightline Watchdog" -Trigger $every5 -Settings $settings `
  -Action (New-ScheduledTaskAction -Execute $ps -Argument "$base `"$here\watchdog.ps1`"")
Write-Output "Registered: Sightline Backend, Sightline ngrok, Sightline Watchdog"
