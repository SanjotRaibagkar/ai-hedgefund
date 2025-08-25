@echo off
REM Activate Poetry Environment for AI Hedge Fund
echo Activating Poetry environment...

REM Set environment variables
set PYTHONPATH=%~dp0..\src
set POETRY_VIRTUALENVS_IN_PROJECT=true

REM Activate Poetry environment
call poetry shell

echo Environment activated!
echo Python path: %PYTHONPATH%
echo Poetry environment: active
