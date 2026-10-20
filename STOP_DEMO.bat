@echo off
setlocal
cd /d "%~dp0"
"%~dp0.acceptance-venv\Scripts\python.exe" -m scripts.acceptance.manage stop
set "suite_exit=%ERRORLEVEL%"
pause
exit /b %suite_exit%
