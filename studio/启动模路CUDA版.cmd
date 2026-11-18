@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s4-cuda\ModelCircuitStudio\ModelCircuitStudio.exe"
if exist "%MODELCIRCUIT_EXE%" (
  start "" "%MODELCIRCUIT_EXE%"
  exit /b 0
)
echo ModelCircuit Studio S4 CUDA has not been built.
echo Run studio/build-s4-desktop.ps1 -Runtime CUDA or extract the complete S4 CUDA portable package.
pause
exit /b 1
