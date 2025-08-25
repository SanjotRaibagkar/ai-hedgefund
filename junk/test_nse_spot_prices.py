#!/usr/bin/env python3
"""
Test NSE Spot Prices
Get current spot prices for NIFTY and BANKNIFTY using different methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def test_nse_spot_prices():
    """Test different methods to get current spot prices."""
    logger.info("🔍 Testing NSE Spot Price Methods")
    
    nse = NseUtils()
    indices = ['NIFTY', 'BANKNIFTY']
    
    for index in indices:
        logger.info(f"\n📊 Testing {index}:")
        
        try:
            # Method 1: Try price_info with .NS suffix
            try:
                price_info = nse.price_info(f"{index}.NS")
                if price_info and 'lastPrice' in price_info:
                    nse_price = float(price_info['lastPrice'])
                    logger.info(f"✅ Method 1 - price_info({index}.NS): ₹{nse_price:,.2f}")
                else:
                    logger.info(f"❌ Method 1 - No price info for {index}.NS")
            except Exception as e:
                logger.info(f"❌ Method 1 - Error: {e}")
            
            # Method 2: Try price_info without .NS suffix
            try:
                price_info = nse.price_info(index)
                if price_info and 'lastPrice' in price_info:
                    nse_price = float(price_info['lastPrice'])
                    logger.info(f"✅ Method 2 - price_info({index}): ₹{nse_price:,.2f}")
                else:
                    logger.info(f"❌ Method 2 - No price info for {index}")
            except Exception as e:
                logger.info(f"❌ Method 2 - Error: {e}")
            
            # Method 3: Try equity_info
            try:
                equity_info = nse.equity_info(f"{index}.NS")
                if equity_info and 'lastPrice' in equity_info:
                    equity_price = float(equity_info['lastPrice'])
                    logger.info(f"✅ Method 3 - equity_info({index}.NS): ₹{equity_price:,.2f}")
                else:
                    logger.info(f"❌ Method 3 - No equity info for {index}.NS")
            except Exception as e:
                logger.info(f"❌ Method 3 - Error: {e}")
            
            # Method 4: Try get_equity_full_list and find the index
            try:
                equity_list = nse.get_equity_full_list()
                if equity_list is not None and not equity_list.empty:
                    # Look for the index in the equity list
                    index_data = equity_list[equity_list['SYMBOL'] == index]
                    if not index_data.empty:
                        if 'LTP' in index_data.columns:
                            ltp_price = float(index_data['LTP'].iloc[0])
                            logger.info(f"✅ Method 4 - Equity list LTP: ₹{ltp_price:,.2f}")
                        else:
                            logger.info(f"❌ Method 4 - No LTP column in equity list")
                    else:
                        logger.info(f"❌ Method 4 - {index} not found in equity list")
                else:
                    logger.info(f"❌ Method 4 - No equity list data")
            except Exception as e:
                logger.info(f"❌ Method 4 - Error: {e}")
            
            # Method 5: Try get_fno_full_list and find the index
            try:
                fno_list = nse.get_fno_full_list()
                if fno_list is not None and not fno_list.empty:
                    # Look for the index in the FNO list
                    index_data = fno_list[fno_list['SYMBOL'] == index]
                    if not index_data.empty:
                        if 'LTP' in index_data.columns:
                            ltp_price = float(index_data['LTP'].iloc[0])
                            logger.info(f"✅ Method 5 - FNO list LTP: ₹{ltp_price:,.2f}")
                        else:
                            logger.info(f"❌ Method 5 - No LTP column in FNO list")
                    else:
                        logger.info(f"❌ Method 5 - {index} not found in FNO list")
                else:
                    logger.info(f"❌ Method 5 - No FNO list data")
            except Exception as e:
                logger.info(f"❌ Method 5 - Error: {e}")
            
            # Method 6: Try to calculate from ATM options (most accurate)
            try:
                options_data = nse.get_live_option_chain(index, indices=True)
                if options_data is not None and not options_data.empty:
                    # Get all strikes
                    strikes = sorted(options_data['Strike_Price'].unique())
                    
                    # Find the strike with highest volume (likely ATM)
                    atm_strikes = []
                    for strike in strikes:
                        strike_data = options_data[options_data['Strike_Price'] == strike]
                        if not strike_data.empty:
                            call_volume = float(strike_data['CALLS_Volume'].iloc[0]) if 'CALLS_Volume' in strike_data.columns else 0
                            put_volume = float(strike_data['PUTS_Volume'].iloc[0]) if 'PUTS_Volume' in strike_data.columns else 0
                            total_volume = call_volume + put_volume
                            atm_strikes.append((strike, total_volume))
                    
                    # Sort by volume and get the highest
                    atm_strikes.sort(key=lambda x: x[1], reverse=True)
                    if atm_strikes:
                        highest_volume_strike = atm_strikes[0][0]
                        logger.info(f"✅ Method 6 - Highest volume strike: ₹{highest_volume_strike:,.2f}")
                        logger.info(f"   (This is likely closest to current spot price)")
                    else:
                        logger.info(f"❌ Method 6 - No volume data available")
                else:
                    logger.info(f"❌ Method 6 - No options data")
            except Exception as e:
                logger.info(f"❌ Method 6 - Error: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error testing {index}: {e}")

if __name__ == "__main__":
    test_nse_spot_prices()
