@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s10\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" (
  echo The S10 ModelCircuit Studio package is unavailable. See studio/native/S10_PROJECT_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_EXE%"
