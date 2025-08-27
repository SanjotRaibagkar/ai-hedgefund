#!/usr/bin/env python3
"""
Fundamental Data Collection Package
Organized collection of fundamental data from NSE for all companies.
"""

__version__ = "1.0.0"
__author__ = "AI Hedge Fund Team"

from .collectors.nse_fundamental_collector import NSEFundamentalCollector, FundamentalData
from .schedulers.fundamental_scheduler import FundamentalScheduler

__all__ = [
    'NSEFundamentalCollector',
    'FundamentalData', 
    'FundamentalScheduler'
]
