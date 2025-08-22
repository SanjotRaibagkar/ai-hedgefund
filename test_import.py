#!/usr/bin/env python3
"""
Simple import test script
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.tools.enhanced_api import get_prices
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}") 