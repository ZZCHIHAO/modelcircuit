@echo off
set "MODELCIRCUIT_PYTHON=%~dp0.venv-cuda\Scripts\pythonw.exe"
if not exist "%MODELCIRCUIT_PYTHON%" (
  echo The CUDA source environment is unavailable. See studio/native/S5_TRANSFORMER_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_PYTHON%" "%~dp0desktop_entry.py" --data-dir "%LOCALAPPDATA%\ModelCircuitStudio-S5-CUDA"
