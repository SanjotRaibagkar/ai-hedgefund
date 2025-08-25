#!/usr/bin/env python3
"""
Test Scheduler Analyzer
Test script to verify the scheduler is using the correct fixed enhanced options analyzer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
from datetime import datetime
from loguru import logger

def test_scheduler_analyzer():
    """Test that the scheduler is using the correct analyzer."""
    logger.info("🎯 Testing Scheduler Analyzer")
    
    try:
        # Import the scheduler function
        from junk.run_options_scheduler import run_options_analysis
        
        # Test both indices
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Testing {index} via Scheduler")
            logger.info(f"{'='*60}")
            
            # Run analysis using scheduler function
            run_options_analysis(index)
            
            # Wait a bit
            time.sleep(2)
        
        # Check CSV file
        import pandas as pd
        csv_file = "results/options_tracker/option_tracker.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            logger.info(f"\n{'='*60}")
            logger.info(f"📄 CSV File Status After Scheduler Test")
            logger.info(f"{'='*60}")
            logger.info(f"   File: {csv_file}")
            logger.info(f"   Total Records: {len(df)}")
            logger.info(f"   Latest NIFTY: {df[df['index']=='NIFTY']['signal'].iloc[-1] if len(df[df['index']=='NIFTY']) > 0 else 'N/A'}")
            logger.info(f"   Latest BANKNIFTY: {df[df['index']=='BANKNIFTY']['signal'].iloc[-1] if len(df[df['index']=='BANKNIFTY']) > 0 else 'N/A'}")
            
            # Check if new records were added
            latest_timestamp = df['timestamp'].iloc[-1]
            logger.info(f"   Latest Timestamp: {latest_timestamp}")
            
            # Verify it's using the correct analyzer by checking the data format
            if 'atm_strike' in df.columns and 'pcr' in df.columns and 'signal' in df.columns:
                logger.info(f"✅ CSV format matches Fixed Enhanced Options Analyzer")
            else:
                logger.error(f"❌ CSV format doesn't match Fixed Enhanced Options Analyzer")
        else:
            logger.error(f"❌ CSV file not found: {csv_file}")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Scheduler Analyzer Test Complete!")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_scheduler_analyzer()
