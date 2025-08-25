#!/usr/bin/env python3
"""
Test NSE API Methods
Verify futures_data, get_option_chain, and get_live_option_chain methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def test_nse_api_methods():
    """Test NSE API methods for futures and options data."""
    logger.info("🧪 Testing NSE API Methods")
    logger.info("=" * 50)
    
    try:
        # Initialize NSE utility
        nse = NseUtils()
        
        # Test indices
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            logger.info(f"\n📊 Testing {index}...")
            
            # 1. Test futures_data method
            logger.info(f"1️⃣ Testing futures_data for {index}...")
            try:
                futures_data = nse.futures_data(index)
                
                if futures_data and isinstance(futures_data, list) and len(futures_data) > 0:
                    logger.info(f"   ✅ Futures data retrieved successfully")
                    logger.info(f"   📊 Number of contracts: {len(futures_data)}")
                    
                    # Show first contract details
                    first_contract = futures_data[0]
                    logger.info(f"   📋 First contract keys: {list(first_contract.keys())}")
                    logger.info(f"   📋 First contract sample: {first_contract}")
                    
                    # Extract price
                    price = None
                    if 'lastPrice' in first_contract:
                        price = float(first_contract['lastPrice'])
                    elif 'ltp' in first_contract:
                        price = float(first_contract['ltp'])
                    elif 'lastTradedPrice' in first_contract:
                        price = float(first_contract['lastTradedPrice'])
                    
                    if price:
                        logger.info(f"   💰 Futures Price: ₹{price:.2f}")
                    else:
                        logger.warning(f"   ⚠️ Could not extract price from futures data")
                        
                else:
                    logger.warning(f"   ⚠️ No futures data for {index}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error in futures_data: {e}")
            
            # 2. Test get_option_chain method
            logger.info(f"2️⃣ Testing get_option_chain for {index}...")
            try:
                options_data = nse.get_option_chain(index, indices=True)
                
                if options_data is not None and not options_data.empty:
                    logger.info(f"   ✅ Options chain retrieved successfully")
                    logger.info(f"   📊 Total records: {len(options_data)}")
                    logger.info(f"   📋 Columns: {list(options_data.columns)}")
                    
                    # Show sample data
                    logger.info(f"   📋 Sample data:")
                    logger.info(f"      {options_data.head(3).to_string()}")
                    
                    # Check for key columns
                    required_cols = ['strikePrice', 'instrumentType']
                    missing_cols = [col for col in required_cols if col not in options_data.columns]
                    if missing_cols:
                        logger.warning(f"   ⚠️ Missing columns: {missing_cols}")
                    else:
                        logger.info(f"   ✅ All required columns present")
                        
                else:
                    logger.warning(f"   ⚠️ No options data for {index}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error in get_option_chain: {e}")
            
            # 3. Test get_live_option_chain method
            logger.info(f"3️⃣ Testing get_live_option_chain for {index}...")
            try:
                live_options_data = nse.get_live_option_chain(index, indices=True)
                
                if live_options_data is not None and not live_options_data.empty:
                    logger.info(f"   ✅ Live options chain retrieved successfully")
                    logger.info(f"   📊 Total records: {len(live_options_data)}")
                    logger.info(f"   📋 Columns: {list(live_options_data.columns)}")
                    
                    # Show sample data
                    logger.info(f"   📋 Sample data:")
                    logger.info(f"      {live_options_data.head(3).to_string()}")
                    
                else:
                    logger.warning(f"   ⚠️ No live options data for {index}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error in get_live_option_chain: {e}")
            
            # 4. Test price_info method as fallback
            logger.info(f"4️⃣ Testing price_info for {index}...")
            try:
                price_info = nse.price_info(index)
                
                if price_info and isinstance(price_info, dict):
                    logger.info(f"   ✅ Price info retrieved successfully")
                    logger.info(f"   📋 Keys: {list(price_info.keys())}")
                    
                    if 'LastTradedPrice' in price_info:
                        price = float(price_info['LastTradedPrice'])
                        logger.info(f"   💰 Last Traded Price: ₹{price:.2f}")
                    else:
                        logger.warning(f"   ⚠️ No LastTradedPrice in price info")
                        
                else:
                    logger.warning(f"   ⚠️ No price info for {index}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error in price_info: {e}")
        
        logger.info(f"\n✅ NSE API methods test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_nse_api_methods()
