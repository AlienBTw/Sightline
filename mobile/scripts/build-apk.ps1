# Build the installable Sightline APK locally (Windows). Run from mobile\:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build-apk.ps1 [-DefaultServer https://<name>.ngrok-free.app] [-Clean] [-UniversalOnly]
# - Generates android\ with `npx expo prebuild` (Continuous Native Generation; android\ is gitignored).
# - Signs with <work>\keys\sightline-release.jks. On first run it creates the keystore and a random
#   password, stored ONLY in that folder (work\ is gitignored; override with -KeyDir).
# - Output: <work>\sightline_release\sightline.apk (universal) and sightline-arm64.apk (phones only, smaller).
#   Log (detached run): mobile_build.log from Get-SightlineLogDir (backend\scripts\_logdir.ps1):
#   $env:SIGHTLINE_LOG_DIR_WIN, else <work>\logs, else $env:LOCALAPPDATA\Sightline\logs.
#   work\ is <repo>\work, or $env:SIGHTLINE_WORK_DIR.
#   If the repo path is too long for the native build, set SIGHTLINE_WORK_DIR or -BuildDir to a short folder.
param(
  [string]$DefaultServer = "",
  [switch]$Clean,
  [string]$KeyDir = "",
  [string]$OutDir = "",
  [string]$BuildDir = "",
  [switch]$UniversalOnly
)
# Native tools write progress to stderr; Windows PowerShell would turn that into terminating errors under
# "Stop", so failures are detected through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$work = if ($env:SIGHTLINE_WORK_DIR) { $env:SIGHTLINE_WORK_DIR } else { Join-Path $repo 'work' }
. "$PSScriptRoot\..\..\backend\scripts\_logdir.ps1"
# Detached runs write mobile_build.log, mobile_build.pid and mobile_build.err.log here.
$logDir = Get-SightlineLogDir
Write-Verbose "mobile build log dir: $logDir"
if (-not $KeyDir) { $KeyDir = Join-Path $work 'keys' }
if (-not $OutDir) { $OutDir = Join-Path $work 'sightline_release' }
if (-not $BuildDir) { $BuildDir = Join-Path $work 'slb' }
$mobile = (Resolve-Path "$PSScriptRoot\..").Path
# The native (CMake/ninja) steps embed the full source path in object paths and fail past Windows'
# 260-character limit when the repo sits in a long folder. A subst drive doesn't help (Node resolves it back
# to the real path), so the project is mirrored to a short real folder and built there. android\ and
# .expo\ in the mirror are kept between runs, so later builds are incremental.
$source = $mobile
if ($mobile.Length -gt 40) {
  $mirror = Join-Path $BuildDir "mobile"
  New-Item -ItemType Directory -Force $mirror | Out-Null
  # Exclude only the top-level android\ and .expo\ (a bare "android" would also skip node_modules\*\android).
  robocopy $source $mirror /MIR /XD "$source\android" "$source\.expo" "$mirror\android" "$mirror\.expo" /MT:16 /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy to $mirror failed ($LASTEXITCODE)" }
  $mobile = $mirror
  Write-Output "building in $mirror (short path mirror of $source)"
}
Set-Location $mobile
try {

# JDK: prefer JDK 17/21 if present, else Android Studio's bundled JBR.
if (-not $env:JAVA_HOME -or -not (Test-Path "$env:JAVA_HOME\bin\java.exe")) {
  $cands = @(Get-ChildItem "C:\Program Files\Eclipse Adoptium","C:\Program Files\Microsoft" -Directory -ErrorAction SilentlyContinue |
             Where-Object Name -match '^jdk-(17|21)') + @(Get-Item "C:\Program Files\Android\Android Studio\jbr" -ErrorAction SilentlyContinue)
  if (-not $cands) { throw "No JDK found. Install one: winget install EclipseAdoptium.Temurin.17.JDK" }
  $env:JAVA_HOME = $cands[0].FullName
}
if (-not $env:ANDROID_HOME) { $env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk" }
$env:PATH = "$env:JAVA_HOME\bin;$env:ANDROID_HOME\platform-tools;$env:PATH"
Write-Output "JAVA_HOME=$env:JAVA_HOME"
cmd /c "`"$env:JAVA_HOME\bin\java.exe`" -version 2>&1" | Select-Object -First 1

# Release keystore (under the work dir, gitignored)
New-Item -ItemType Directory -Force $KeyDir | Out-Null
$ks = Join-Path $KeyDir "sightline-release.jks"
$props = Join-Path $KeyDir "sightline-release.properties"
if (-not (Test-Path $ks)) {
  $pw = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
  & "$env:JAVA_HOME\bin\keytool.exe" -genkeypair -v -storetype PKCS12 -keystore $ks -alias sightline -keyalg RSA -keysize 2048 `
    -validity 10000 -storepass $pw -keypass $pw -dname "CN=Sightline, O=Sightline, C=TN" 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $ks)) { throw "keytool failed" }
  Set-Content -Path $props -Value @("storePassword=$pw", "keyPassword=$pw", "keyAlias=sightline") -Encoding ascii
  Write-Output "created release keystore $ks (passwords in $props)"
}
$kv = @{}
Get-Content $props | ForEach-Object { if ($_ -match '^\s*([^=#]+)=(.*)$') { $kv[$matches[1].Trim()] = $matches[2] } }
$env:ORG_GRADLE_PROJECT_SIGHTLINE_STORE_FILE = $ks
$env:ORG_GRADLE_PROJECT_SIGHTLINE_STORE_PASSWORD = $kv['storePassword']
$env:ORG_GRADLE_PROJECT_SIGHTLINE_KEY_ALIAS = $kv['keyAlias']
$env:ORG_GRADLE_PROJECT_SIGHTLINE_KEY_PASSWORD = $kv['keyPassword']

# Optional build-time default server (not a secret; can be changed on the login screen)
if ($DefaultServer) { $env:SIGHTLINE_DEFAULT_SERVER = $DefaultServer }
$env:CI = "1"
$env:NODE_ENV = "production"

$pre = @("expo", "prebuild", "--platform", "android", "--no-install")
if ($Clean -or -not (Test-Path "android\app\build.gradle")) { $pre += "--clean" }
& npx @pre
if ($LASTEXITCODE -ne 0) { throw "expo prebuild failed" }

# Use the SDK's newest CMake when present: its ninja (1.12) handles paths over 260 characters (with
# Windows LongPathsEnabled); the default CMake 3.22's ninja 1.10 does not. Install with:
#   sdkmanager "cmake;3.31.6"
$cmakeDir = Get-ChildItem "$env:ANDROID_HOME\cmake" -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^3\.(2[89]|3\d)\.' } | Sort-Object { [version]$_.Name } | Select-Object -Last 1
if ($cmakeDir) {
  $lp = "android\local.properties"
  $lines = @(Get-Content $lp -ErrorAction SilentlyContinue | Where-Object { $_ -notmatch '^cmake\.dir=' })
  $lines += "cmake.dir=" + ($cmakeDir.FullName -replace '\\', '\\' -replace ':', '\:')
  Set-Content $lp $lines -Encoding ascii
  Write-Output "using CMake $($cmakeDir.Name)"
}
# Two APKs: a universal one (phones + x86_64 emulators) and a smaller arm64-only one (almost every phone
# since 2017). Each Gradle run reuses the other's cached native objects.
$variants = @(
  @{ Abis = "arm64-v8a,armeabi-v7a,x86_64"; Name = "sightline.apk" },
  @{ Abis = "arm64-v8a"; Name = "sightline-arm64.apk" }
)
if ($UniversalOnly) { $variants = @($variants[0]) }
New-Item -ItemType Directory -Force $OutDir | Out-Null
foreach ($v in $variants) {
  Push-Location android
  try {
    & .\gradlew.bat assembleRelease --no-daemon "-Dorg.gradle.jvmargs=-Xmx4g" "-PreactNativeArchitectures=$($v.Abis)"
    if ($LASTEXITCODE -ne 0) { throw "gradle assembleRelease failed ($($v.Abis))" }
  } finally { Pop-Location }
  $apk = Get-ChildItem "android\app\build\outputs\apk\release\*.apk" | Sort-Object LastWriteTime | Select-Object -Last 1
  $dest = Join-Path $OutDir $v.Name
  Copy-Item $apk.FullName $dest -Force
  $out = Get-Item $dest
  $hash = (Get-FileHash $out.FullName -Algorithm SHA256).Hash
  Set-Content -Path "$dest.sha256" -Value "$hash  $($v.Name)" -Encoding ascii
  Write-Output ("APK: {0} ({1:N1} MB) sha256 {2}" -f $out.FullName, ($out.Length / 1MB), $hash)
}
} finally {
  Set-Location $PSScriptRoot
}
