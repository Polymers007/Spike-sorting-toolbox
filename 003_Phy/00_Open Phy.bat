@echo off
chcp 65001 >nul
setlocal
title Open original Phy - default TemplateGUI
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Program\Open_Phy.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Phy was not started successfully. See the message above.
    pause
)
endlocal & exit /b %EXIT_CODE%
