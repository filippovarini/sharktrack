@echo off
title SharkTrack - Marine Video Analysis
cd /d "%~dp0"

echo.
echo ==========================================
echo       SharkTrack - Marine Video Analysis
echo ==========================================
echo.

REM Check Python using py launcher (more reliable on Windows)
py --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.12.10:
    echo   https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
    echo.
    echo Also disable App Execution Aliases in Windows Settings:
    echo   Settings -^> Apps -^> Advanced app settings -^> App execution aliases
    echo   Turn OFF both python.exe entries
    echo.
    pause
    exit /b
)
echo [OK] Python found

REM Create venv if needed
if not exist "sharktrack-env" (
    echo.
    echo Creating virtual environment...
    py -m venv sharktrack-env
    echo [OK] Virtual environment created
)

REM Activate venv
echo.
echo Activating virtual environment...
call sharktrack-env\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies (this may take 5-10 minutes on first run)...
echo.
echo NOTE: PyTorch and Ultralytics are large packages (several GB).
echo       The install may appear to pause - this is normal, please wait.
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed!
    echo.
    pause
    exit /b
)

REM Create directories
if not exist "models" mkdir models
if not exist "templates" mkdir templates
if not exist "static" mkdir static

REM Network test (triggers firewall prompt)
echo.
echo Checking network access...
echo (If Windows Firewall asks, click ALLOW)
echo.
py -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 5000)); s.close(); print('[OK] Port 5000 available')"

echo.
echo ==========================================
echo Starting SharkTrack server...
echo ==========================================
echo.
echo Open this URL in your browser:
echo.
echo    http://localhost:5000
echo.
echo (Keep this window open - it runs the server)
echo (Press Ctrl+C to stop)
echo ==========================================
echo.

REM Start the server
py start_sharktrack.py

echo.
echo Server stopped.
echo.
pause
