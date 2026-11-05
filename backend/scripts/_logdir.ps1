# Shared Windows log directory. Dot-source, then call Get-SightlineLogDir:
#   . "$PSScriptRoot\_logdir.ps1"
# $env:SIGHTLINE_LOG_DIR_WIN wins when it is set.
# Otherwise the candidate is <work>\logs, where work is $env:SIGHTLINE_WORK_DIR
# or <repo>\work. If that candidate exists but is a reparse point Windows cannot
# open (Test-Path fails, or Get-ChildItem throws), or it cannot be created or
# written (a temp file is created and removed), the fallback is
# $env:LOCALAPPDATA\Sightline\logs. The chosen directory is created with
# New-Item -Force and returned.
# Captured at dot-source time so the function still finds this file later.
$script:SightlineLogHelperRoot = $PSScriptRoot
if (-not $script:SightlineLogHelperRoot -and $MyInvocation.MyCommand.Path) {
  $script:SightlineLogHelperRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

function Get-SightlineLogDir {
  function Test-CandidateUsable([string]$Path) {
    $isReparse = $false
    $haveAttrs = $false
    try {
      $attrs = [System.IO.File]::GetAttributes($Path)
      $haveAttrs = $true
      $isReparse = [bool]($attrs -band [System.IO.FileAttributes]::ReparsePoint)
    } catch {}

    $testOk = $false
    $testThrew = $false
    try { $testOk = [bool](Test-Path -LiteralPath $Path -ErrorAction Stop) } catch { $testThrew = $true }

    $listThrew = $false
    if ($testOk) {
      try {
        Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop | Select-Object -First 1 | Out-Null
      } catch { $listThrew = $true }
    }

    # Exists, but Windows cannot open it (broken reparse point, or a directory we cannot list).
    if ($isReparse -and ($testThrew -or -not $testOk -or $listThrew)) { return $false }
    if ($haveAttrs -and ($testThrew -or $listThrew)) { return $false }
    if ($testThrew) { return $false }

    $probe = $null
    try {
      New-Item -ItemType Directory -Force -Path $Path -ErrorAction Stop | Out-Null
      $probe = Join-Path $Path ('.sightline-logdir-probe-' + [guid]::NewGuid().ToString('n'))
      [System.IO.File]::WriteAllText($probe, 'ok')
      return $true
    } catch {
      return $false
    } finally {
      if ($probe) { Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue }
    }
  }

  if ($env:SIGHTLINE_LOG_DIR_WIN) {
    $chosen = $env:SIGHTLINE_LOG_DIR_WIN
  } else {
    if (-not $script:SightlineLogHelperRoot) {
      throw 'Get-SightlineLogDir: dot-source backend/scripts/_logdir.ps1 first'
    }
    $repo = (Resolve-Path (Join-Path $script:SightlineLogHelperRoot '..\..')).Path
    $work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $repo 'work' }
    $candidate = Join-Path $work 'logs'
    if (Test-CandidateUsable $candidate) {
      $chosen = $candidate
    } else {
      if (-not $env:LOCALAPPDATA) {
        throw "log dir $candidate is not usable and LOCALAPPDATA is unset"
      }
      $chosen = Join-Path $env:LOCALAPPDATA 'Sightline\logs'
    }
  }
  New-Item -ItemType Directory -Force -Path $chosen -ErrorAction Stop | Out-Null
  return $chosen
}
