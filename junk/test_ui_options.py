#!/usr/bin/env python3
"""
Test UI Options Analysis
Test if the UI will show correct spot prices from futures data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nsedata.NseUtility import NseUtils
from loguru import logger

def test_ui_options_analysis():
    """Test the same logic that the UI uses for options analysis."""
    logger.info("🎯 Testing UI Options Analysis Logic")
    
    nse = NseUtils()
    indices = ['NIFTY', 'BANKNIFTY']
    
    for index in indices:
        logger.info(f"\n📊 Testing {index}:")
        
        try:
            # Get options data
            options_data = nse.get_live_option_chain(index, indices=True)
            
            if options_data is not None and not options_data.empty:
                # Get spot price from futures data (same as updated UI)
                try:
                    futures_data = nse.futures_data(index, indices=True)
                    if futures_data is not None and not futures_data.empty:
                        # Get the first row (current month contract) for spot price
                        current_price = float(futures_data['lastPrice'].iloc[0])
                        logger.info(f"✅ Using futures data: ₹{current_price:,.2f}")
                    else:
                        # Fallback to options data if futures fails
                        strikes = sorted(options_data['Strike_Price'].unique())
                        current_price = float(strikes[len(strikes)//2])
                        logger.info(f"⚠️ Using options fallback: ₹{current_price:,.2f}")
                except Exception as e:
                    # Fallback to options data if futures fails
                    strikes = sorted(options_data['Strike_Price'].unique())
                    current_price = float(strikes[len(strikes)//2])
                    logger.info(f"⚠️ Using options fallback (error): ₹{current_price:,.2f}")
                
                # Find ATM strike
                strikes = sorted(options_data['Strike_Price'].unique())
                atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                
                logger.info(f"🎯 ATM Strike: ₹{atm_strike:,.2f}")
                logger.info(f"📊 This is what the UI will show!")
                
            else:
                logger.error(f"❌ No options data for {index}")
                
        except Exception as e:
            logger.error(f"❌ Error testing {index}: {e}")

if __name__ == "__main__":
    test_ui_options_analysis()
