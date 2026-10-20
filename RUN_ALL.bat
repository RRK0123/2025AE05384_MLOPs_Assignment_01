@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_acceptance.ps1"
set "suite_exit=%ERRORLEVEL%"
echo.
echo Complete suite finished with exit code %suite_exit%.
echo Open the printed START_HERE.html. Keep demo services running until evidence is saved.
pause
exit /b %suite_exit%
