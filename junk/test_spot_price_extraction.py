#!/usr/bin/env python3
"""
Test Spot Price Extraction
Check how to properly extract current spot price from NSE options data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def test_spot_price_extraction():
    """Test different methods to extract spot price."""
    logger.info("🔍 Testing Spot Price Extraction Methods")
    
    nse = NseUtils()
    indices = ['NIFTY', 'BANKNIFTY']
    
    for index in indices:
        logger.info(f"\n📊 Testing {index}:")
        
        try:
            # Get options data
            options_data = nse.get_live_option_chain(index, indices=True)
            
            if options_data is not None and not options_data.empty:
                logger.info(f"✅ Got options data for {index}")
                logger.info(f"   Columns: {list(options_data.columns)}")
                logger.info(f"   Shape: {options_data.shape}")
                
                # Method 1: Check if there's an underlyingValue column
                if 'underlyingValue' in options_data.columns:
                    spot_price = float(options_data['underlyingValue'].iloc[0])
                    logger.info(f"✅ Method 1 - underlyingValue: ₹{spot_price:,.2f}")
                else:
                    logger.info("❌ Method 1 - No 'underlyingValue' column")
                
                # Method 2: Check if there's a spot price column
                spot_columns = [col for col in options_data.columns if 'spot' in col.lower() or 'price' in col.lower()]
                logger.info(f"   Potential spot columns: {spot_columns}")
                
                # Method 3: Current wrong method
                strikes = sorted(options_data['Strike_Price'].unique())
                wrong_price = float(strikes[len(strikes)//2])
                logger.info(f"❌ Method 3 - Current wrong method (middle strike): ₹{wrong_price:,.2f}")
                
                # Method 4: Try to get from NSE price info
                try:
                    price_info = nse.price_info(index)
                    if price_info and 'lastPrice' in price_info:
                        nse_price = float(price_info['lastPrice'])
                        logger.info(f"✅ Method 4 - NSE price_info: ₹{nse_price:,.2f}")
                    else:
                        logger.info("❌ Method 4 - No price info available")
                except Exception as e:
                    logger.info(f"❌ Method 4 - Error: {e}")
                
                # Method 5: Try futures data
                try:
                    futures_data = nse.futures_data(index)
                    if futures_data is not None and not futures_data.empty:
                        logger.info(f"✅ Method 5 - Got futures data")
                        logger.info(f"   Futures columns: {list(futures_data.columns)}")
                        if 'lastPrice' in futures_data.columns:
                            futures_price = float(futures_data['lastPrice'].iloc[0])
                            logger.info(f"   Futures price: ₹{futures_price:,.2f}")
                    else:
                        logger.info("❌ Method 5 - No futures data")
                except Exception as e:
                    logger.info(f"❌ Method 5 - Error: {e}")
                
                # Show sample data
                logger.info(f"\n📋 Sample options data (first 3 rows):")
                logger.info(options_data.head(3).to_string())
                
            else:
                logger.error(f"❌ No options data for {index}")
                
        except Exception as e:
            logger.error(f"❌ Error testing {index}: {e}")

if __name__ == "__main__":
    test_spot_price_extraction()
