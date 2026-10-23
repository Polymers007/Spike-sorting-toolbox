@echo off
setlocal
title Phy Curation Manager
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Curation_Manager.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Curation Manager was not started successfully. See the message above.
    pause
)
endlocal & exit /b %EXIT_CODE%
