@echo off
REM Install Options Data Collection Service for Windows
echo Installing Options Data Collection Service...

REM Set environment variables
set SERVICE_NAME=AIHedgeFundOptionsCollector
set SERVICE_DISPLAY_NAME=AI Hedge Fund Options Data Collector
set SERVICE_DESCRIPTION=Automated options data collection service for NIFTY and BANKNIFTY
set PYTHON_PATH=C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\ai-hedge-fund-Dot02FSf-py3.13\Scripts\python.exe
set SCRIPT_PATH=%~dp0..\start_options_collector.py
set WORKING_DIR=%~dp0..

REM Create the service using sc command
sc create "%SERVICE_NAME%" binPath= "\"%PYTHON_PATH%\" \"%SCRIPT_PATH%\"" DisplayName= "%SERVICE_DISPLAY_NAME%" start= auto

REM Set service description
sc description "%SERVICE_NAME%" "%SERVICE_DESCRIPTION%"

REM Set service to start automatically
sc config "%SERVICE_NAME%" start= auto

REM Set service to restart on failure
sc failure "%SERVICE_NAME%" reset= 86400 actions= restart/60000/restart/60000/restart/60000

echo Service installed successfully!
echo Service name: %SERVICE_NAME%
echo To start: sc start %SERVICE_NAME%
echo To stop: sc stop %SERVICE_NAME%
echo To remove: sc delete %SERVICE_NAME%
