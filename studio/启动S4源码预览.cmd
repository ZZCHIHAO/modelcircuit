@echo off
set "MLFORGE_PREVIEW_PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%MLFORGE_PREVIEW_PYTHON%" (
  echo The existing CPU development environment is unavailable.
  echo See studio/native/S4_COMPONENT_WORKBENCH_VERIFICATION.md.
  pause
  exit /b 1
)
start "" "%MLFORGE_PREVIEW_PYTHON%" "%~dp0desktop_entry.py" --data-dir "%LOCALAPPDATA%\MLForgeStudio-S4Preview-CPU"
