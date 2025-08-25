#!/usr/bin/env python3
"""
Monitor Options Data Collection
Simple script to monitor the options data collection status
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.downloaders.options_chain_collector import OptionsChainCollector
from loguru import logger


def monitor_options_data():
    """Monitor options data collection status."""
    try:
        logger.info("📊 Monitoring Options Data Collection")
        
        # Initialize collector
        collector = OptionsChainCollector()
        
        # Check current status
        logger.info("📅 Trading day check...")
        is_trading_day = collector._is_trading_day()
        logger.info(f"   Is trading day: {is_trading_day}")
        
        logger.info("⏰ Market hours check...")
        is_market_hours = collector._is_market_hours()
        logger.info(f"   Is market hours: {is_market_hours}")
        
        # Get daily summaries
        logger.info("📊 Daily data summary...")
        for index in ['NIFTY', 'BANKNIFTY']:
            summary = collector.get_daily_summary(index)
            if summary:
                logger.info(f"   {index}:")
                logger.info(f"     Total records: {summary.get('total_records', 0):,}")
                logger.info(f"     Unique strikes: {summary.get('unique_strikes', 0)}")
                logger.info(f"     Avg spot price: ₹{summary.get('avg_spot_price', 0):,.2f}")
                logger.info(f"     PCR: {summary.get('avg_pcr', 0):.3f}")
            else:
                logger.info(f"   {index}: No data for today")
        
        # Get recent data sample
        logger.info("📊 Recent data sample...")
        for index in ['NIFTY', 'BANKNIFTY']:
            recent_data = collector.get_recent_data(index, minutes=60)
            if not recent_data.empty:
                logger.info(f"   {index} last hour: {len(recent_data)} records")
                if len(recent_data) > 0:
                    latest = recent_data.iloc[0]
                    logger.info(f"     Latest spot: ₹{latest['spot_price']:,.2f}")
                    logger.info(f"     Latest ATM strike: {latest['atm_strike']:,.0f}")
            else:
                logger.info(f"   {index}: No recent data")
        
        logger.info("✅ Monitoring completed!")
        
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    monitor_options_data()
