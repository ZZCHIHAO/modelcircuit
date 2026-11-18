param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
$studioRoot = $PSScriptRoot
if (-not $PythonPath) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $PythonPath = $pythonCommand.Source }
}
if (-not $PythonPath) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $PythonPath = $bundledPython }
}
if (-not $PythonPath) { throw 'Provide Python 3.11 or 3.12 with -PythonPath.' }
$venvPython = Join-Path $studioRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $PythonPath -m venv (Join-Path $studioRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment creation failed.' }
}
& $venvPython -m pip install filelock networkx jinja2 fsspec sympy==1.13.1 'typing_extensions>=4.10' --index-url https://pypi.org/simple --timeout 60
if ($LASTEXITCODE -ne 0) { throw 'PyTorch supporting dependencies failed to install.' }
& $venvPython -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu --no-deps --timeout 60
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
Push-Location (Join-Path $studioRoot 'frontend')
try {
    & npm.cmd ci --prefer-offline --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
} finally { Pop-Location }
Write-Output 'Dependencies ready. Run studio/start.ps1.'

