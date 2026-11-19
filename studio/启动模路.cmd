@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s9\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" (
  echo The S9 ModelCircuit Studio package is unavailable. See studio/native/S9_INFERENCE_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_EXE%"
