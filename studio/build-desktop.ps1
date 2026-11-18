param([switch]$InstallDependencies, [ValidateSet('CPU','CUDA')][string]$Runtime='CPU')
$ErrorActionPreference = 'Stop'
$studioPath = $PSScriptRoot
$repositoryPath = Split-Path -Parent $studioPath
$environmentName = if ($Runtime -eq 'CUDA') { '.venv-cuda' } else { '.venv' }
$desktopPython = Join-Path $studioPath ($environmentName + '\Scripts\python.exe')
if (-not (Test-Path -LiteralPath $desktopPython)) { throw 'Run studio/setup.ps1 first to prepare Python and CPU PyTorch.' }
if ($InstallDependencies) {
    & $desktopPython -m pip install -r (Join-Path $studioPath 'requirements-desktop.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Desktop dependency installation failed.' }
}
$env:PYTHONPATH = $studioPath + ';' + (Join-Path $studioPath 'backend')
& $desktopPython -m unittest discover -s (Join-Path $studioPath 'native\tests') -v
if ($LASTEXITCODE -ne 0) { throw 'Native tests failed; build stopped.' }
$targetName = if ($Runtime -eq 'CUDA') { 'desktop-cuda' } else { 'desktop' }
$distributionPath = Join-Path $studioPath ('dist\' + $targetName)
$buildPath = Join-Path $studioPath ('build\' + $targetName)
& $desktopPython -m PyInstaller --noconfirm --distpath $distributionPath --workpath $buildPath (Join-Path $studioPath 'native\MLForgeStudio.spec')
if ($LASTEXITCODE -ne 0) { throw 'Windows executable build failed.' }
$desktopExe = Join-Path $distributionPath 'MLForgeStudio\MLForgeStudio.exe'
if (-not (Test-Path -LiteralPath $desktopExe)) { throw 'Build did not produce the expected executable.' }
Write-Output ('Windows application: ' + $desktopExe)
Write-Output 'Keep MLForgeStudio.exe together with its _internal folder. This portable build needs neither a Python nor a Node.js command to launch.'
