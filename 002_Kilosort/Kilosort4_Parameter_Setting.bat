@echo off
chcp 65001 >nul
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Start_Kilosort.ps1" -Mode Parameters
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
