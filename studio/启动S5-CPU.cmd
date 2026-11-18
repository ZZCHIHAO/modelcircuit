@echo off
set "MODELCIRCUIT_PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%MODELCIRCUIT_PYTHON%" (
  echo The CPU source environment is unavailable. See studio/native/S5_TRANSFORMER_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_PYTHON%" "%~dp0desktop_entry.py" --data-dir "%LOCALAPPDATA%\ModelCircuitStudio-S5-CPU"
