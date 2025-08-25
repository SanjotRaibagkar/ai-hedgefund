#!/usr/bin/env python3
"""
Start Auto Options Data Collector
Script to start the auto options collector with proper configuration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.downloaders.auto_options_collector import AutoOptionsCollector
from loguru import logger


def main():
    """Start the auto options collector."""
    try:
        logger.info("🚀 Starting Auto Options Data Collector")
        logger.info("📅 This will automatically start at 9:30 AM and stop at 3:30 PM")
        logger.info("📊 Monitoring and status tracking enabled")
        logger.info("🛑 Press Ctrl+C to stop")
        
        # Initialize and start the auto collector
        auto_collector = AutoOptionsCollector()
        
        # Run with scheduling (auto start/stop)
        auto_collector.run_with_schedule()
        
    except KeyboardInterrupt:
        logger.info("🛑 Auto Options Collector stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in Auto Options Collector: {e}")
        raise


if __name__ == "__main__":
    main()
