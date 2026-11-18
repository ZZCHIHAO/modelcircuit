param([string]$RepositoryName = 'modelcircuit')
$ErrorActionPreference = 'Stop'
if ($RepositoryName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Use a valid GitHub repository name.' }
$repositoryPath = Split-Path -Parent $PSScriptRoot
$ghCommand = Get-Command gh -ErrorAction SilentlyContinue
$ghPath = if ($ghCommand) { $ghCommand.Source } else {
    Get-ChildItem -LiteralPath (Join-Path $env:LOCALAPPDATA 'MLForgeStudio-build-tools') -Recurse -Filter gh.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ghPath) { throw 'GitHub CLI is missing. Install the official GitHub CLI first.' }
& $ghPath auth status
if ($LASTEXITCODE -ne 0) { throw 'Sign in using studio/登录GitHub.cmd before uploading. Never paste a token into chat.' }
$account = & $ghPath api user | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $account.login) { throw 'Cannot verify the signed-in personal GitHub account.' }
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
$gitPath = if ($gitCommand) { $gitCommand.Source } else {
    $bundledGit = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
    if (Test-Path -LiteralPath $bundledGit) { $bundledGit }
}
if (-not $gitPath) { throw 'Install Git for Windows or make git available on PATH.' }
$env:PATH = (Split-Path -Parent $gitPath) + ';' + $env:PATH
Set-Location -LiteralPath $repositoryPath
if (-not (Test-Path -LiteralPath '.git')) {
    & $gitPath init -b main
    if ($LASTEXITCODE -ne 0) { throw 'Git initialization failed.' }
}
& $gitPath config --local user.name $account.login
& $gitPath config --local user.email ($account.id.ToString() + '+' + $account.login + '@users.noreply.github.com')
& $gitPath config --local core.quotepath false
& $gitPath config --local core.autocrlf false
& $gitPath add --all
if ($LASTEXITCODE -ne 0) { throw 'Cannot stage source files.' }
$stagedPaths = & $gitPath ls-files --cached
foreach ($relativePath in $stagedPaths) {
    if ($relativePath -match '(^|/)(node_modules|\.venv|dist|build|runs|logs)(/|$)|(^|/)\.env($|\.)') { throw ('Runtime/private state must not be uploaded: ' + $relativePath) }
    $candidate = Get-Item -LiteralPath (Join-Path $repositoryPath $relativePath)
    if ($candidate.Length -gt 50MB) { throw ('Oversized source file: ' + $relativePath) }
    if ($candidate.Extension -in @('.py','.ps1','.cmd','.js','.jsx','.mjs','.json','.toml','.yaml','.yml','.md','.txt') -and $candidate.Length -lt 1MB) {
        $candidateText = Get-Content -LiteralPath $candidate.FullName -Raw
        if ($candidateText -match '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b') { throw ('Potential credential detected; upload stopped: ' + $relativePath) }
    }
}
& $gitPath diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    & $gitPath commit -m 'Add ModelCircuit Studio native AI architecture workbench'
    if ($LASTEXITCODE -ne 0) { throw 'Git commit failed.' }
}
$slug = $account.login + '/' + $RepositoryName
$existingRepository = & $ghPath repo view $slug --json isPrivate,url 2>$null
if ($LASTEXITCODE -ne 0) {
    & $ghPath repo create $slug --private --description 'ModelCircuit Studio: native visual AI architecture and experiment workbench; S4 source milestone'
    if ($LASTEXITCODE -ne 0) { throw 'Private repository creation failed.' }
} else {
    $knownOrigin = & $gitPath remote get-url origin 2>$null
    if ($LASTEXITCODE -ne 0 -or $knownOrigin -notmatch ('github\.com[:/]' + [regex]::Escape($slug) + '(\.git)?$')) {
        throw 'That repository already exists and is not this checkout origin; choose another name to avoid touching unrelated work.'
    }
}
$remote = & $ghPath repo view $slug --json isPrivate,url | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $remote.isPrivate) { throw 'Repository privacy could not be confirmed; source upload stopped.' }
& $gitPath remote get-url origin 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { & $gitPath remote add origin ('https://github.com/' + $slug + '.git') }
$credentialHelper = '!"' + ($ghPath -replace '\\','/') + '" auth git-credential'
& $gitPath -c 'credential.helper=' -c ('credential.helper=' + $credentialHelper) push --set-upstream origin main
if ($LASTEXITCODE -ne 0) { throw 'Source push failed; no force push was attempted.' }
$sharing = & $ghPath api ('repos/' + $slug + '/collaborators') | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Source uploaded, but collaborator visibility could not be verified.' }
$others = @($sharing | Where-Object { $_.login -ne $account.login })
if ($others.Count -ne 0) { throw 'Source uploaded privately, but unexpected collaborators were found. Review access before sharing.' }
Write-Output ('Verified private source repository, no added collaborators: ' + $remote.url)
