@echo off
set "MODELCIRCUIT_EXE=%~dp0dist\s4-cpu\ModelCircuitStudio\ModelCircuitStudio.exe"
if exist "%MODELCIRCUIT_EXE%" (
  start "" "%MODELCIRCUIT_EXE%"
  exit /b 0
)
echo ModelCircuit Studio S4 CPU has not been built.
echo Run studio/build-s4-desktop.ps1 -Runtime CPU or extract the complete S4 CPU portable package.
pause
exit /b 1
