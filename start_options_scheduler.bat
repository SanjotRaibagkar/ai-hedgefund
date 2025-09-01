@echo off
echo Starting Options Scheduler...
echo.
echo This will run the options analysis scheduler
echo The scheduler will automatically stop at 3:30 PM
echo.
echo Press Ctrl+C to stop manually
echo.
cd /d "%~dp0"
poetry run python junk/run_options_scheduler.py
pause
