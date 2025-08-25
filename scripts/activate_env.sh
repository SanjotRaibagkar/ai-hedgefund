#!/bin/bash
# Activate Poetry Environment for AI Hedge Fund

echo "Activating Poetry environment..."

# Set environment variables
export PYTHONPATH="$(pwd)/src"
export POETRY_VIRTUALENVS_IN_PROJECT=true

# Activate Poetry environment
poetry shell

echo "Environment activated!"
echo "Python path: $PYTHONPATH"
echo "Poetry environment: active"
