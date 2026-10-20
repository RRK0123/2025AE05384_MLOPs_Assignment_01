@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.acceptance-venv\Scripts\python.exe" (
  "%~dp0.acceptance-venv\Scripts\python.exe" -m scripts.acceptance.manage collect
) else (
  where py >nul 2>&1
  if errorlevel 1 (
    python -m scripts.acceptance.manage collect
  ) else (
    py -3.13 -m scripts.acceptance.manage collect
  )
)
set "suite_exit=%ERRORLEVEL%"
pause
exit /b %suite_exit%
