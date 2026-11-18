@echo off
set "MLFORGE_CUDA_EXE=%~dp0dist\desktop-cuda\MLForgeStudio\MLForgeStudio.exe"
if exist "%MLFORGE_CUDA_EXE%" (
  start "" "%MLFORGE_CUDA_EXE%"
  exit /b 0
)
set "MLFORGE_CUDA_PYTHON=%~dp0.venv-cuda\Scripts\pythonw.exe"
if exist "%MLFORGE_CUDA_PYTHON%" (
  start "" "%MLFORGE_CUDA_PYTHON%" "%~dp0desktop_entry.py"
  exit /b 0
)
echo The independent CUDA environment is not installed. See studio/native/README.md.
pause
