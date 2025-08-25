#!/usr/bin/env python3
"""
Debug Futures Data Extraction
Test why the futures data extraction is not working in the options tracker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def debug_futures_extraction():
    """Debug the futures data extraction process."""
    logger.info("🔍 Debugging Futures Data Extraction")
    
    nse = NseUtils()
    indices = ['NIFTY', 'BANKNIFTY']
    
    for index in indices:
        logger.info(f"\n📊 Testing {index}:")
        
        try:
            # Test futures data extraction (same as in options tracker)
            futures_data = nse.futures_data(index, indices=True)
            
            if futures_data is not None and not futures_data.empty:
                logger.info(f"✅ Got futures data for {index}")
                logger.info(f"   Shape: {futures_data.shape}")
                logger.info(f"   Columns: {list(futures_data.columns)}")
                logger.info(f"   First row: {futures_data.iloc[0].to_dict()}")
                
                # Extract spot price (same logic as options tracker)
                current_price = float(futures_data['lastPrice'].iloc[0])
                logger.info(f"✅ Extracted spot price: ₹{current_price:,.2f}")
                
            else:
                logger.error(f"❌ No futures data for {index}")
                
        except Exception as e:
            logger.error(f"❌ Error getting futures data for {index}: {e}")
            
        # Also test options data
        try:
            options_data = nse.get_live_option_chain(index, indices=True)
            
            if options_data is not None and not options_data.empty:
                logger.info(f"✅ Got options data for {index}")
                logger.info(f"   Shape: {options_data.shape}")
                logger.info(f"   Strike prices: {sorted(options_data['Strike_Price'].unique())[:5]}...")
                
                # Test the fallback method
                strikes = sorted(options_data['Strike_Price'].unique())
                fallback_price = float(strikes[len(strikes)//2])
                logger.info(f"⚠️ Fallback price (middle strike): ₹{fallback_price:,.2f}")
                
            else:
                logger.error(f"❌ No options data for {index}")
                
        except Exception as e:
            logger.error(f"❌ Error getting options data for {index}: {e}")

if __name__ == "__main__":
    debug_futures_extraction()

