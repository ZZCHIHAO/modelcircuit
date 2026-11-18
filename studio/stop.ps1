$ErrorActionPreference = 'Stop'
$studioRoot = $PSScriptRoot
$pidFile = Join-Path $studioRoot '.processes.json'
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Output 'No launcher-owned processes recorded.'; exit 0 }
$saved = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
foreach ($property in $saved.PSObject.Properties) {
    $record = $property.Value
    $processInfo = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$record.id) -ErrorAction SilentlyContinue
    if ($processInfo -and $processInfo.CommandLine -and $processInfo.CommandLine.IndexOf([string]$record.command, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        # Children are stopped only when both parent identity and this project's Python path match.
        Get-CimInstance Win32_Process -Filter ('ParentProcessId = ' + [int]$record.id) -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_.ExecutablePath -and $_.ExecutablePath.StartsWith((Join-Path $studioRoot '.venv'), [System.StringComparison]::OrdinalIgnoreCase)) {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
        }
        Stop-Process -Id ([int]$record.id) -Force -ErrorAction SilentlyContinue
    }
}
Write-Output 'Launcher-owned local services stopped.'

