@echo off
chcp 65001 >nul
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Start_Kilosort.ps1" -Mode Batch %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
  echo Batch pass finished.
) else (
  echo Batch sorting stopped with exit code %EXIT_CODE%.
)
echo You may close this window, or press any key.
pause >nul
endlocal & exit /b %EXIT_CODE%
