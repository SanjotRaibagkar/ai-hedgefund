#!/usr/bin/env python3
"""
Basic Options Collector Test
Simple test to verify the options collector works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.downloaders.options_chain_collector import OptionsChainCollector
from loguru import logger


def main():
    """Test basic options collector functionality."""
    try:
        logger.info("🧪 Testing Options Chain Collector - Basic Test")
        
        # Initialize collector
        collector = OptionsChainCollector()
        
        # Test 1: Check if it's a trading day
        is_trading_day = collector._is_trading_day()
        logger.info(f"📅 Is trading day: {is_trading_day}")
        
        # Test 2: Check market hours
        is_market_hours = collector._is_market_hours()
        logger.info(f"⏰ Is market hours: {is_market_hours}")
        
        # Test 3: Get trading holidays
        holidays = collector._get_trading_holidays()
        logger.info(f"📅 Trading holidays count: {len(holidays)}")
        
        # Test 4: Try to get futures spot price for NIFTY
        logger.info("📊 Testing NIFTY futures spot price...")
        spot_price = collector._get_futures_spot_price('NIFTY')
        if spot_price:
            logger.info(f"✅ NIFTY spot price: ₹{spot_price:,.2f}")
        else:
            logger.warning("⚠️ Could not get NIFTY spot price")
        
        # Test 5: Try to get futures spot price for BANKNIFTY
        logger.info("📊 Testing BANKNIFTY futures spot price...")
        spot_price = collector._get_futures_spot_price('BANKNIFTY')
        if spot_price:
            logger.info(f"✅ BANKNIFTY spot price: ₹{spot_price:,.2f}")
        else:
            logger.warning("⚠️ Could not get BANKNIFTY spot price")
        
        logger.info("✅ Basic test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
