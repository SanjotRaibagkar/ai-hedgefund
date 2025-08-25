@echo off
REM Setup Windows Task Scheduler for Options Data Collection
echo Setting up Windows Task Scheduler...

REM Set variables
set TASK_NAME=AIHedgeFundOptionsCollector
set TASK_DESCRIPTION=Start options data collection service at 9:30 AM on trading days
set PYTHON_PATH=C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\ai-hedge-fund-Dot02FSf-py3.13\Scripts\python.exe
set SCRIPT_PATH=%~dp0..\start_options_collector.py
set WORKING_DIR=%~dp0..

REM Create the scheduled task
schtasks /create /tn "%TASK_NAME%" /tr "\"%PYTHON_PATH%\" \"%SCRIPT_PATH%\"" /sc daily /st 09:30 /sd %date% /ru SYSTEM /f

REM Set task description
schtasks /change /tn "%TASK_NAME%" /description "%TASK_DESCRIPTION%"

REM Enable the task
schtasks /change /tn "%TASK_NAME%" /enable

echo Task scheduled successfully!
echo Task name: %TASK_NAME%
echo Start time: 9:30 AM daily
echo To run manually: schtasks /run /tn "%TASK_NAME%"
echo To delete: schtasks /delete /tn "%TASK_NAME%" /f
echo To view: schtasks /query /tn "%TASK_NAME%"
