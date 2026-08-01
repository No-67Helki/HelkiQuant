param([string]$ProjectPython = "python")

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RuntimeRoot = Join-Path $RepoRoot ".runtime\rqsdk310"
$Requirements = Join-Path $PSScriptRoot "configs\requirements-rqdata.txt"

if (-not (Get-Command $ProjectPython -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $ProjectPython"
}
if (-not (Test-Path -LiteralPath (Join-Path $RuntimeRoot "Scripts\python.exe"))) {
    & $ProjectPython -m venv $RuntimeRoot
}
& (Join-Path $RuntimeRoot "Scripts\python.exe") -m pip install -r $Requirements
& (Join-Path $RuntimeRoot "Scripts\rqsdk.exe") version
