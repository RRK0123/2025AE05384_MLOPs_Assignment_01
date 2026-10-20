@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_deployment.ps1" -CollectOnly
set "deployment_exit=%ERRORLEVEL%"
echo.
echo Log collection finished with exit code %deployment_exit%.
pause
exit /b %deployment_exit%
