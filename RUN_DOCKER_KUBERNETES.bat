@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_deployment.ps1"
set "deployment_exit=%ERRORLEVEL%"
echo.
echo Launcher finished with exit code %deployment_exit%.
echo Read the report path printed above. Keep the tunnel window open during your demo.
pause
exit /b %deployment_exit%
