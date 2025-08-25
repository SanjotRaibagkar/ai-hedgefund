#!/usr/bin/env python3
"""
Test Futures Spot Price
Check how to extract spot price from futures_data method.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def test_futures_spot_price():
    """Test futures_data method to get spot prices."""
    logger.info("🔍 Testing Futures Data for Spot Prices")
    
    nse = NseUtils()
    indices = ['NIFTY', 'BANKNIFTY']
    
    for index in indices:
        logger.info(f"\n📊 Testing {index}:")
        
        try:
            # Get futures data with indices=True
            futures_data = nse.futures_data(index, indices=True)
            
            if futures_data is not None and not futures_data.empty:
                logger.info(f"✅ Got futures data for {index}")
                logger.info(f"   Columns: {list(futures_data.columns)}")
                logger.info(f"   Shape: {futures_data.shape}")
                
                # Show first few rows
                logger.info(f"\n📋 Sample futures data (first 3 rows):")
                logger.info(futures_data.head(3).to_string())
                
                # Look for spot price columns
                spot_columns = [col for col in futures_data.columns if 'spot' in col.lower() or 'price' in col.lower() or 'ltp' in col.lower()]
                logger.info(f"\n🎯 Potential spot price columns: {spot_columns}")
                
                # Try to find the spot price
                if 'LTP' in futures_data.columns:
                    # Get the first row (usually the current month contract)
                    ltp = float(futures_data['LTP'].iloc[0])
                    logger.info(f"✅ Found LTP: ₹{ltp:,.2f}")
                elif 'lastPrice' in futures_data.columns:
                    last_price = float(futures_data['lastPrice'].iloc[0])
                    logger.info(f"✅ Found lastPrice: ₹{last_price:,.2f}")
                elif 'Close' in futures_data.columns:
                    close_price = float(futures_data['Close'].iloc[0])
                    logger.info(f"✅ Found Close: ₹{close_price:,.2f}")
                else:
                    logger.info("❌ No clear spot price column found")
                    logger.info("Available columns with 'price' in name:")
                    for col in spot_columns:
                        if 'price' in col.lower():
                            value = futures_data[col].iloc[0]
                            logger.info(f"   {col}: {value}")
                
            else:
                logger.error(f"❌ No futures data for {index}")
                
        except Exception as e:
            logger.error(f"❌ Error testing {index}: {e}")

if __name__ == "__main__":
    test_futures_spot_price()
