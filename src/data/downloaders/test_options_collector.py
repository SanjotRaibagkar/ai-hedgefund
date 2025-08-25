#!/usr/bin/env python3
"""
Test Options Chain Data Collector
Test script to verify options data collection functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import pandas as pd
from datetime import datetime
from loguru import logger

from .options_chain_collector import OptionsChainCollector


def test_options_collector():
    """Test the options chain collector functionality."""
    logger.info("🧪 Testing Options Chain Data Collector")
    
    try:
        # Initialize collector
        collector = OptionsChainCollector()
        
        # Test 1: Check trading day
        logger.info("📅 Testing trading day check...")
        is_trading_day = collector._is_trading_day()
        logger.info(f"   Is trading day: {is_trading_day}")
        
        # Test 2: Check market hours
        logger.info("⏰ Testing market hours check...")
        is_market_hours = collector._is_market_hours()
        logger.info(f"   Is market hours: {is_market_hours}")
        
        # Test 3: Get trading holidays
        logger.info("📅 Testing trading holidays...")
        holidays = collector._get_trading_holidays()
        logger.info(f"   Trading holidays count: {len(holidays)}")
        if holidays:
            logger.info(f"   Sample holidays: {holidays[:5]}")
        
        # Test 4: Test futures spot price for NIFTY
        logger.info("📊 Testing futures spot price for NIFTY...")
        spot_price = collector._get_futures_spot_price('NIFTY')
        if spot_price:
            logger.info(f"   NIFTY spot price: ₹{spot_price:,.2f}")
        else:
            logger.warning("   ⚠️ Could not get NIFTY spot price")
        
        # Test 5: Test futures spot price for BANKNIFTY
        logger.info("📊 Testing futures spot price for BANKNIFTY...")
        spot_price = collector._get_futures_spot_price('BANKNIFTY')
        if spot_price:
            logger.info(f"   BANKNIFTY spot price: ₹{spot_price:,.2f}")
        else:
            logger.warning("   ⚠️ Could not get BANKNIFTY spot price")
        
        # Test 6: Test options data collection for NIFTY
        logger.info("📊 Testing options data collection for NIFTY...")
        success = collector.collect_data_for_index('NIFTY')
        logger.info(f"   NIFTY collection success: {success}")
        
        # Test 7: Test options data collection for BANKNIFTY
        logger.info("📊 Testing options data collection for BANKNIFTY...")
        success = collector.collect_data_for_index('BANKNIFTY')
        logger.info(f"   BANKNIFTY collection success: {success}")
        
        # Test 8: Get recent data
        logger.info("📊 Testing recent data retrieval...")
        for index in ['NIFTY', 'BANKNIFTY']:
            recent_data = collector.get_recent_data(index, minutes=60)
            logger.info(f"   {index} recent records: {len(recent_data)}")
            if not recent_data.empty:
                logger.info(f"   {index} sample data:")
                logger.info(f"   {recent_data.head(3).to_string()}")
        
        # Test 9: Get daily summary
        logger.info("📊 Testing daily summary...")
        for index in ['NIFTY', 'BANKNIFTY']:
            summary = collector.get_daily_summary(index)
            if summary:
                logger.info(f"   {index} daily summary:")
                for key, value in summary.items():
                    logger.info(f"     {key}: {value}")
            else:
                logger.info(f"   {index}: No data for today")
        
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


def test_single_collection():
    """Test a single data collection run."""
    logger.info("🎯 Testing single data collection run")
    
    try:
        collector = OptionsChainCollector()
        
        # Run collection for all indices
        success = collector.collect_all_data()
        
        if success:
            logger.info("✅ Single collection run successful")
            
            # Show summary
            for index in ['NIFTY', 'BANKNIFTY']:
                summary = collector.get_daily_summary(index)
                if summary:
                    logger.info(f"📊 {index} today: {summary.get('total_records', 0)} records")
        else:
            logger.warning("⚠️ Single collection run had issues")
            
    except Exception as e:
        logger.error(f"❌ Single collection test failed: {e}")
        raise


def test_database_connection():
    """Test database connection and table structure."""
    logger.info("🗄️ Testing database connection")
    
    try:
        collector = OptionsChainCollector()
        
        # Test connection
        connection = collector.connection
        
        # Check if options_chain_data table exists
        result = connection.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='options_chain_data'
        """).fetchone()
        
        if result:
            logger.info("✅ options_chain_data table exists")
            
            # Get table schema
            schema = connection.execute("PRAGMA table_info(options_chain_data)").fetchdf()
            logger.info(f"📋 Table schema ({len(schema)} columns):")
            for _, row in schema.iterrows():
                logger.info(f"   {row['name']}: {row['type']}")
        else:
            logger.error("❌ options_chain_data table does not exist")
            
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        raise


def main():
    """Main test function."""
    logger.info("🚀 Starting Options Chain Collector Tests")
    
    try:
        # Test 1: Database connection
        test_database_connection()
        
        # Test 2: Basic functionality
        test_options_collector()
        
        # Test 3: Single collection
        test_single_collection()
        
        logger.info("🎉 All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
