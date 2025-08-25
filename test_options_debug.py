#!/usr/bin/env python3
"""
Debug Options Data Collection
Test script to debug options data collection issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.downloaders.options_chain_collector import OptionsChainCollector
from src.nsedata.NseUtility import NseUtils
from loguru import logger


def debug_options_data():
    """Debug options data collection."""
    try:
        logger.info("🔍 Debugging Options Data Collection")
        
        # Initialize NSE utility
        nse = NseUtils()
        
        # Test 1: Get futures data for NIFTY
        logger.info("📊 Testing NIFTY futures data...")
        futures_data = nse.futures_data('NIFTY', indices=True)
        if futures_data is not None and not futures_data.empty:
            logger.info(f"✅ NIFTY futures data shape: {futures_data.shape}")
            logger.info(f"✅ NIFTY futures columns: {list(futures_data.columns)}")
            logger.info(f"✅ NIFTY futures sample:")
            logger.info(futures_data.head(3).to_string())
        else:
            logger.error("❌ No NIFTY futures data")
        
        # Test 2: Get options chain data for NIFTY
        logger.info("📊 Testing NIFTY options chain data...")
        options_data = nse.get_live_option_chain('NIFTY', indices=True)
        if options_data is not None and not options_data.empty:
            logger.info(f"✅ NIFTY options data shape: {options_data.shape}")
            logger.info(f"✅ NIFTY options columns: {list(options_data.columns)}")
            logger.info(f"✅ NIFTY options sample:")
            logger.info(options_data.head(3).to_string())
            
            # Check for specific columns
            call_columns = [col for col in options_data.columns if 'CALL' in col.upper()]
            put_columns = [col for col in options_data.columns if 'PUT' in col.upper()]
            logger.info(f"✅ CALL columns: {call_columns}")
            logger.info(f"✅ PUT columns: {put_columns}")
        else:
            logger.error("❌ No NIFTY options data")
        
        # Test 3: Test BANKNIFTY
        logger.info("📊 Testing BANKNIFTY futures data...")
        futures_data = nse.futures_data('BANKNIFTY', indices=True)
        if futures_data is not None and not futures_data.empty:
            logger.info(f"✅ BANKNIFTY futures data shape: {futures_data.shape}")
            logger.info(f"✅ BANKNIFTY futures columns: {list(futures_data.columns)}")
        else:
            logger.error("❌ No BANKNIFTY futures data")
        
        logger.info("📊 Testing BANKNIFTY options chain data...")
        options_data = nse.get_live_option_chain('BANKNIFTY', indices=True)
        if options_data is not None and not options_data.empty:
            logger.info(f"✅ BANKNIFTY options data shape: {options_data.shape}")
            logger.info(f"✅ BANKNIFTY options columns: {list(options_data.columns)}")
        else:
            logger.error("❌ No BANKNIFTY options data")
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_options_data()
