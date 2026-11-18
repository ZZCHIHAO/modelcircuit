@echo off
set "MLFORGE_DESKTOP=%~dp0dist\desktop\MLForgeStudio\MLForgeStudio.exe"
if exist "%MLFORGE_DESKTOP%" (
  start "" "%MLFORGE_DESKTOP%"
  exit /b 0
)
echo The desktop executable has not been built. Run studio/build-desktop.ps1 first.
pause
