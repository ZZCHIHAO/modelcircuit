@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s14b\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" set "MODELCIRCUIT_EXE=%~dp0dist\s14\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" set "MODELCIRCUIT_EXE=%~dp0dist\s13\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" set "MODELCIRCUIT_EXE=%~dp0dist\s12\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" set "MODELCIRCUIT_EXE=%~dp0dist\s11\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" set "MODELCIRCUIT_EXE=%~dp0dist\s10\ModelCircuitStudio\ModelCircuitStudio.exe"
if not exist "%MODELCIRCUIT_EXE%" (
  echo The ModelCircuit Studio package is unavailable. See studio/native/S14B_EXPERIMENT_GUIDE.md.
  pause
  exit /b 1
)
start "" "%MODELCIRCUIT_EXE%" %*
