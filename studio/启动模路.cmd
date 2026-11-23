@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s12\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" set "MODELCIRCUIT_EXE=%~dp0dist\s11\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" set "MODELCIRCUIT_EXE=%~dp0dist\s10\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" (
  echo The ModelCircuit Studio package is unavailable. See studio/native/S12_REMOTE_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_EXE%"
