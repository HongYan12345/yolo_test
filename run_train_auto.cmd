@echo off
setlocal

cd /d "%~dp0"

where pwsh >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_auto.ps1" %*
    goto :exit
)

where powershell >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_auto.ps1" %*
    goto :exit
)

echo [ERROR] Neither pwsh nor powershell was found.
echo Please install PowerShell or run train_auto.ps1 manually.
exit /b 1

:exit
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Finished with exit code %EXIT_CODE%.
if %EXIT_CODE% NEQ 0 (
    echo.
    echo Training failed or required inputs are missing.
    echo Check the messages above.
)
pause
exit /b %EXIT_CODE%
