@echo off
setlocal
cd /d "%~dp0\.."

echo [Edit With My Voice] Windows setup

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.11 from https://www.python.org/downloads/windows/
  pause
  exit /b 1
)

if not exist .venv (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Could not create a Python 3.11 virtual environment. Make sure Python 3.11 is installed.
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg was not found. Trying winget install...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  echo If FFmpeg was just installed, close and reopen this terminal before running again.
)

echo Setup complete.
pause
