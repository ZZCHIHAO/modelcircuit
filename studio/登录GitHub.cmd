@echo off
setlocal
set "MLFORGE_GH="
for /r "%LOCALAPPDATA%\MLForgeStudio-build-tools" %%G in (gh.exe) do if exist "%%G" set "MLFORGE_GH=%%G"
if not defined MLFORGE_GH (
  echo GitHub CLI was not found. Install it from cli.github.com first.
  pause
  exit /b 1
)
"%MLFORGE_GH%" auth login --hostname github.com --git-protocol https --web
echo.
"%MLFORGE_GH%" auth status
pause
