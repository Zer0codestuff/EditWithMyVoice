@echo off
setlocal
cd /d "%~dp0\.."

if not exist .venv (
  call scripts\setup_windows.bat
  if errorlevel 1 (
    echo Setup failed. Fix the error above and run again.
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m edit_with_my_voice.app
pause
