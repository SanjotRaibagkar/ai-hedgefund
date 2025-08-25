#!/usr/bin/env python3
"""
Start Options Data Collection Service
Simple script to start the options chain data collector
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.downloaders.options_chain_collector import OptionsChainCollector
from loguru import logger


def main():
    """Start the options data collection service."""
    try:
        logger.info("🚀 Starting Options Data Collection Service")
        logger.info("📊 This will collect NIFTY and BANKNIFTY options data every minute")
        logger.info("⏰ Trading hours: 9:30 AM - 3:30 PM IST")
        logger.info("📅 Only on trading days (excluding weekends and holidays)")
        logger.info("🛑 Press Ctrl+C to stop")
        
        # Initialize and start collector
        collector = OptionsChainCollector()
        collector.start_scheduler()
        
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
