# Builds dist\FrameAnchor.zip - the ready-to-send package - from this repo.
#
# The bundled binaries are not kept in git. This downloads the exact files the
# app ships with and refuses to continue unless each one matches its pinned
# SHA256:
#   bin\uv.exe                                   uv 0.13.8 (Astral, GitHub release)
#   wheels\numpy-2.5.4-cp312-cp312-win_amd64.whl (PyPI)
#   wheels\openvr-2.12.1401-py3-none-any.whl     (PyPI)
#
# Usage (from the repo root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File build_release.ps1 [-SkipTests]

param([switch]$SkipTests)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root  = $PSScriptRoot
$cache = Join-Path $root "build\cache"
$dist  = Join-Path $root "dist"
New-Item -ItemType Directory -Force $cache, $dist | Out-Null

$uvVersion = "0.13.8"
$uvSha     = "03fa174894b9d0cbe08762d40d1c2c64d10e875bd56f1a1e334ee1e00d60cdbe"
$wheels = @(
    @{ Package = "numpy";  Version = "2.5.4";       File = "numpy-2.5.4-cp312-cp312-win_amd64.whl";
       Sha = "a7b1b6353e36a7e50de2973a38d705c88ee93adcf120673cee7f45a4a3fa223a" },
    @{ Package = "openvr"; Version = "2.12.1401";   File = "openvr-2.12.1401-py3-none-any.whl";
       Sha = "ef761cef3162843a8025a7812d5e244fd94ab3a992ca8f5ef0d1b0188ad56cb3" }
)

function Test-Sha($path, $sha) {
    (Test-Path $path) -and ((Get-FileHash $path -Algorithm SHA256).Hash.ToLower() -eq $sha)
}

# --- uv.exe -------------------------------------------------------------------
$uvExe = Join-Path $cache "uv.exe"
if (-not (Test-Sha $uvExe $uvSha)) {
    Write-Host "Downloading uv $uvVersion..."
    $zip = Join-Path $cache "uv.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
    $x = Join-Path $cache "uv-x"
    if (Test-Path $x) { Remove-Item -Recurse -Force $x }
    Expand-Archive $zip $x
    Copy-Item (Get-ChildItem -Recurse $x -Filter uv.exe | Select-Object -First 1).FullName $uvExe -Force
    Remove-Item -Recurse -Force $x, $zip
    if (-not (Test-Sha $uvExe $uvSha)) { throw "uv.exe does not match its pinned SHA256 - refusing to build." }
}

# --- wheels -------------------------------------------------------------------
foreach ($w in $wheels) {
    $dest = Join-Path $cache $w.File
    if (Test-Sha $dest $w.Sha) { continue }
    Write-Host "Downloading $($w.File)..."
    $meta = Invoke-RestMethod "https://pypi.org/pypi/$($w.Package)/$($w.Version)/json" -UseBasicParsing
    $url = ($meta.urls | Where-Object { $_.filename -eq $w.File } | Select-Object -First 1).url
    if (-not $url) { throw "PyPI has no file named $($w.File)" }
    Invoke-WebRequest $url -OutFile $dest -UseBasicParsing
    if (-not (Test-Sha $dest $w.Sha)) { throw "$($w.File) does not match its pinned SHA256 - refusing to build." }
}

# --- tests ----------------------------------------------------------------------
if (-not $SkipTests) {
    Write-Host "Running tests..."
    $t = Join-Path $root "build\test-run"
    if (Test-Path $t) { Remove-Item -Recurse -Force $t }
    New-Item -ItemType Directory -Force $t | Out-Null
    Copy-Item (Join-Path $root "FrameAnchor\app\*.py"), (Join-Path $root "tests\*.py") $t
    Push-Location $t
    try {
        & $uvExe run -q --no-project --managed-python --python 3.12 --find-links $cache `
            --with openvr --with numpy --with pytest -m pytest -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw "Tests failed - not building." }
    } finally { Pop-Location }
}

# --- package ----------------------------------------------------------------------
$stage = Join-Path $root "build\stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$pkg = Join-Path $stage "FrameAnchor"
New-Item -ItemType Directory -Force (Join-Path $pkg "app"), (Join-Path $pkg "bin"), (Join-Path $pkg "wheels") | Out-Null
Copy-Item (Join-Path $root "FrameAnchor\Start FrameAnchor.bat"), (Join-Path $root "FrameAnchor\READ ME FIRST.txt") $pkg
Copy-Item (Join-Path $root "FrameAnchor\app\*.py") (Join-Path $pkg "app")
Copy-Item $uvExe (Join-Path $pkg "bin")
foreach ($w in $wheels) { Copy-Item (Join-Path $cache $w.File) (Join-Path $pkg "wheels") }

$out = Join-Path $dist "FrameAnchor.zip"
if (Test-Path $out) { Remove-Item -Force $out }
Push-Location $stage
try { tar.exe -a -c -f $out FrameAnchor } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "Couldn't create the zip." }
$h = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
Write-Host "Built $out"
Write-Host "SHA256 $h"
