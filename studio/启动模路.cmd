@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s7\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" (
  echo The ModelCircuit Studio package is unavailable. See studio/native/S7_RESEARCH_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_EXE%"
