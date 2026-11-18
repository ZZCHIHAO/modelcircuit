@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s6-cuda\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" (
  echo The S6 CUDA package is unavailable. See studio/native/S6_REGRESSION_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_EXE%"
