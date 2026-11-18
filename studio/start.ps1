param([switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$studioRoot = $PSScriptRoot
$pythonPath = Join-Path $studioRoot '.venv\Scripts\python.exe'
$frontendRoot = Join-Path $studioRoot 'frontend'
$vitePath = Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'
$serverPath = Join-Path $studioRoot 'backend\server.py'
$logDir = Join-Path $studioRoot 'logs'
$pidFile = Join-Path $studioRoot '.processes.json'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'studio/.venv is missing. Follow studio/README.md to prepare Python dependencies.' }
if (-not (Test-Path -LiteralPath $vitePath)) { throw 'Frontend dependencies are missing. Run npm install in studio/frontend.' }
$nodePath = (Get-Command node.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
function ApiReady {
    try { return ((Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 2).version -eq '0.1.0') } catch { return $false }
}
function UiReady {
    try { return ((Invoke-WebRequest 'http://127.0.0.1:4173/' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) } catch { return $false }
}
function AssertPortFree([int]$port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect('127.0.0.1', $port)
        throw ('Port ' + $port + ' is occupied by an unexpected service.')
    } catch [System.Net.Sockets.SocketException] {
        return
    } finally { $client.Dispose() }
}
$processes = @{}
if (Test-Path -LiteralPath $pidFile) {
    try {
        $saved = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
        foreach ($property in $saved.PSObject.Properties) { $processes[$property.Name] = $property.Value }
    } catch {}
}
if (-not (ApiReady)) {
    AssertPortFree 8765
    $apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @(('"' + $serverPath + '"')) -WorkingDirectory $studioRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'api.log') -RedirectStandardError (Join-Path $logDir 'api.error.log') -PassThru
    $processes.api = @{ id = $apiProcess.Id; command = $serverPath }
}
if (-not (UiReady)) {
    AssertPortFree 4173
    $uiProcess = Start-Process -FilePath $nodePath -ArgumentList @(('"' + $vitePath + '"'), '--host', '127.0.0.1', '--port', '4173', '--strictPort') -WorkingDirectory $frontendRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'ui.log') -RedirectStandardError (Join-Path $logDir 'ui.error.log') -PassThru
    $processes.ui = @{ id = $uiProcess.Id; command = $vitePath }
}
$processes | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pidFile -Encoding UTF8
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ((ApiReady) -and (UiReady)) {
        Write-Output 'ML Forge Studio is ready: http://127.0.0.1:4173/'
        if ($OpenBrowser) { Start-Process 'http://127.0.0.1:4173/' }
        exit 0
    }
    Start-Sleep -Milliseconds 500
}
throw ('Service startup did not finish. See logs in ' + $logDir)

