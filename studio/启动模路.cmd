@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s8\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" (
  echo The S8 ModelCircuit Studio package is unavailable. See studio/native/S8_SEGMENTATION_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_EXE%"
