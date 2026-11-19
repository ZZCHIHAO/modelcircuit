param([ValidateSet('CPU','CUDA','Unified')][string]$Runtime='CPU', [ValidateSet('S4','S6','S7','S8')][string]$Stage='S4')
$ErrorActionPreference = 'Stop'
if ($Stage -in ('S6','S7','S8') -and -not $PSBoundParameters.ContainsKey('Runtime')) {
    $Runtime = 'Unified'
}
if (($Stage -in ('S6','S7','S8')) -ne ($Runtime -eq 'Unified')) {
    throw 'Use -Stage S8 for the current unified desktop package. CPU and CUDA build variants are only available for the historical S4 stage.'
}
$studioPath = $PSScriptRoot
$repositoryPath = Split-Path -Parent $studioPath
$environmentName = if ($Runtime -in ('CUDA','Unified')) { '.venv-cuda' } else { '.venv' }
$desktopPython = Join-Path $studioPath ($environmentName + '\Scripts\python.exe')
if (-not (Test-Path -LiteralPath $desktopPython)) { throw 'Prepare the selected environment before building. This script does not install dependencies.' }
$targetName = if ($Runtime -eq 'Unified') { $Stage.ToLowerInvariant() } else { 's4-' + $Runtime.ToLowerInvariant() }
$distributionPath = Join-Path $studioPath ('dist\' + $targetName)
$buildPath = Join-Path $studioPath ('build\' + $targetName)
$buildManifestPath = Join-Path $buildPath 'release-build.json'
$oldPythonPath = $env:PYTHONPATH
$oldManifest = $env:MODELCIRCUIT_RELEASE_MANIFEST
$oldEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = $studioPath + ';' + (Join-Path $studioPath 'backend')
    $env:PYTHONIOENCODING = 'utf-8'
    & $desktopPython -m native.release_metadata --repository $repositoryPath --runtime $Runtime --stage $Stage --output $buildManifestPath
    if ($LASTEXITCODE -ne 0) { throw 'Release identity or runtime variant check failed.' }
    if ($Stage -eq 'S4') {
        & $desktopPython -m unittest discover -s (Join-Path $studioPath 'native\tests') -v
        if ($LASTEXITCODE -ne 0) { throw 'Native regression failed; build stopped.' }
    }
    $env:MODELCIRCUIT_RELEASE_MANIFEST = $buildManifestPath
    & $desktopPython -m PyInstaller --noconfirm --distpath $distributionPath --workpath $buildPath (Join-Path $studioPath 'native\ModelCircuitStudio.spec')
    if ($LASTEXITCODE -ne 0) { throw "$Stage desktop build failed." }
    $desktopExe = Join-Path $distributionPath 'ModelCircuitStudio\ModelCircuitStudio.exe'
    if (-not (Test-Path -LiteralPath $desktopExe)) { throw 'Expected desktop executable is missing.' }
    Copy-Item -LiteralPath $buildManifestPath -Destination (Join-Path (Split-Path -Parent $desktopExe) 'release-build.json')
    Write-Output ($Stage + ' ' + $Runtime + ' application: ' + $desktopExe)
    Write-Output 'Keep the complete ModelCircuitStudio directory, including _internal and release-build.json.'
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:MODELCIRCUIT_RELEASE_MANIFEST = $oldManifest
    $env:PYTHONIOENCODING = $oldEncoding
}
