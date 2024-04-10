@echo off
setlocal
:: This script is to run SharkTrack on windows without access to the command line

:: Add the virtual environment's Python interpreter to the PATH
set "PATH=..\venv\bin\python;%PATH%"

:: Navigate to the directory containing app.py if necessary
cd ..\

:: Run the application
python app.py

:: End of script
endlocal
