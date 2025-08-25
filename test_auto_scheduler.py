#!/usr/bin/env python3
"""
Test Auto Market Hours Scheduler
Test script to verify the auto market hours scheduler works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
from datetime import datetime
from loguru import logger

def test_auto_scheduler():
    """Test the auto market hours scheduler."""
    logger.info("🎯 Testing Auto Market Hours Scheduler")
    
    try:
        from junk.auto_market_hours_scheduler import AutoMarketHoursScheduler
        
        # Initialize scheduler
        scheduler = AutoMarketHoursScheduler()
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 Testing Scheduler Functions")
        logger.info(f"{'='*60}")
        
        # Test trading day check
        is_trading_day = scheduler._is_trading_day()
        logger.info(f"✅ Trading day check: {is_trading_day}")
        
        # Test market hours check
        is_market_hours = scheduler._is_market_hours()
        logger.info(f"✅ Market hours check: {is_market_hours}")
        
        # Test should run analysis
        should_run = scheduler._should_run_analysis()
        logger.info(f"✅ Should run analysis: {should_run}")
        
        # Test analysis function
        logger.info(f"\n{'='*60}")
        logger.info("📊 Testing Analysis Function")
        logger.info(f"{'='*60}")
        
        # Run analysis for both indices
        for index in ['NIFTY', 'BANKNIFTY']:
            logger.info(f"🎯 Testing {index} analysis...")
            scheduler.run_options_analysis(index)
            time.sleep(2)
        
        # Check CSV file
        import pandas as pd
        csv_file = "results/options_tracker/option_tracker.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            logger.info(f"\n{'='*60}")
            logger.info(f"📄 CSV File Status After Auto Scheduler Test")
            logger.info(f"{'='*60}")
            logger.info(f"   File: {csv_file}")
            logger.info(f"   Total Records: {len(df)}")
            logger.info(f"   Latest NIFTY: {df[df['index']=='NIFTY']['signal'].iloc[-1] if len(df[df['index']=='NIFTY']) > 0 else 'N/A'}")
            logger.info(f"   Latest BANKNIFTY: {df[df['index']=='BANKNIFTY']['signal'].iloc[-1] if len(df[df['index']=='BANKNIFTY']) > 0 else 'N/A'}")
            
            # Check if new records were added
            latest_timestamp = df['timestamp'].iloc[-1]
            logger.info(f"   Latest Timestamp: {latest_timestamp}")
        else:
            logger.error(f"❌ CSV file not found: {csv_file}")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Auto Market Hours Scheduler Test Complete!")
        logger.info(f"{'='*60}")
        
        # Show scheduler info
        logger.info(f"\n{'='*60}")
        logger.info("📋 Scheduler Information")
        logger.info(f"{'='*60}")
        logger.info(f"   Market Hours: {scheduler.market_open} - {scheduler.market_close} IST")
        logger.info(f"   Analysis Interval: {scheduler.analysis_interval} minutes")
        logger.info(f"   Trading Day Check: {is_trading_day}")
        logger.info(f"   Market Hours Check: {is_market_hours}")
        logger.info(f"   Should Run Analysis: {should_run}")
        
        # Show setup instructions
        logger.info(f"\n{'='*60}")
        logger.info("🚀 Setup Instructions for Tomorrow")
        logger.info(f"{'='*60}")
        logger.info("To set up automatic startup for tomorrow:")
        logger.info("1. Run as Administrator: scripts\\setup_auto_options_scheduler.bat")
        logger.info("2. The scheduler will start automatically at system startup")
        logger.info("3. It will wait for market hours (9:30 AM) before running analysis")
        logger.info("4. Analysis will run every 15 minutes during market hours")
        logger.info("5. It will stop at market close (3:30 PM)")
        logger.info("6. This repeats daily on trading days only")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_auto_scheduler()
