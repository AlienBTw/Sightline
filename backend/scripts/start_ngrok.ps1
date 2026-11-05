# Expose the backend over HTTPS with ngrok (free dev domain). The authtoken lives in the ngrok config
# (ngrok config add-authtoken ...) or the NGROK_AUTHTOKEN env var - never in this repo.
#   $env:SIGHTLINE_NGROK_URL = "https://<your-dev-domain>.ngrok-free.app"   # optional; default = your dev domain
param([int]$Port = 8000)
. "$PSScriptRoot\_logdir.ps1"
$ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $ngrok) { Write-Output "ngrok not found on PATH (winget install ngrok.ngrok)"; exit 1 }
if (Get-Process ngrok -ErrorAction SilentlyContinue) { Write-Output "ngrok already running"; exit 0 }
$logDir = Get-SightlineLogDir
$log = Join-Path $logDir "ngrok.log"
if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).Length -gt 20MB) {
  Move-Item -Force -LiteralPath $log "$log.1"
}
$nargs = @("http", "$Port", "--log", $log, "--log-format", "json")
if ($env:SIGHTLINE_NGROK_URL) { $nargs += @("--url", $env:SIGHTLINE_NGROK_URL) }
$p = Start-Process -FilePath $ngrok -ArgumentList $nargs -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 4
try {
  $t = Invoke-RestMethod "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 5
  Write-Output "ngrok pid $($p.Id): $($t.tunnels.public_url -join ', ')"
} catch { Write-Output "ngrok started (pid $($p.Id)); local API not answering yet, see $log" }
