@echo off
REM Setup Auto Market Hours Options Scheduler for Windows Task Scheduler
echo Setting up Auto Market Hours Options Scheduler...

REM Set variables
set TASK_NAME=AIHedgeFundAutoOptionsScheduler
set TASK_DESCRIPTION=Auto market hours options analysis scheduler (9:30 AM - 3:30 PM on trading days)
set PYTHON_PATH=C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\ai-hedge-fund-Dot02FSf-py3.13\Scripts\python.exe
set SCRIPT_PATH=%~dp0..\junk\auto_market_hours_scheduler.py
set WORKING_DIR=%~dp0..

REM Create the scheduled task to start at system startup
schtasks /create /tn "%TASK_NAME%" /tr "\"%PYTHON_PATH%\" \"%SCRIPT_PATH%\" --mode auto" /sc onstart /ru SYSTEM /f

REM Set task description
schtasks /change /tn "%TASK_NAME%" /description "%TASK_DESCRIPTION%"

REM Set working directory
schtasks /change /tn "%TASK_NAME%" /s /ru SYSTEM /rp "" /f

REM Enable the task
schtasks /change /tn "%TASK_NAME%" /enable

echo.
echo ========================================
echo Auto Market Hours Options Scheduler Setup Complete!
echo ========================================
echo.
echo Task Details:
echo   Name: %TASK_NAME%
echo   Description: %TASK_DESCRIPTION%
echo   Python: %PYTHON_PATH%
echo   Script: %SCRIPT_PATH%
echo   Working Dir: %WORKING_DIR%
echo.
echo Scheduler Features:
echo   ✅ Starts automatically at system startup
echo   ✅ Runs during market hours (9:30 AM - 3:30 PM IST)
echo   ✅ Respects trading days (excludes weekends and holidays)
echo   ✅ Analysis every 15 minutes during market hours
echo   ✅ Uses Fixed Enhanced Options Analyzer
echo   ✅ Creates option_tracker.csv with accurate data
echo.
echo Commands:
echo   To run manually: schtasks /run /tn "%TASK_NAME%"
echo   To stop: schtasks /end /tn "%TASK_NAME%"
echo   To delete: schtasks /delete /tn "%TASK_NAME%" /f
echo   To view: schtasks /query /tn "%TASK_NAME%"
echo.
echo Logs will be saved to: logs/auto_options_scheduler.log
echo CSV file will be saved to: results/options_tracker/option_tracker.csv
echo.
echo The scheduler will automatically:
echo   1. Start at system startup
echo   2. Wait for market hours (9:30 AM)
echo   3. Run analysis every 15 minutes during market hours
echo   4. Stop analysis at market close (3:30 PM)
echo   5. Repeat daily on trading days only
echo.
pause
