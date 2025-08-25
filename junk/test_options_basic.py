#!/usr/bin/env python3
"""
Basic Options Test
Test basic options analysis functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def test_basic_options():
    """Test basic options analysis."""
    logger.info("🧪 Testing Basic Options Analysis")
    logger.info("=" * 40)
    
    try:
        nse = NseUtils()
        
        # Test NIFTY
        logger.info("📊 Testing NIFTY options...")
        
        # Get options data
        options_data = nse.get_live_option_chain('NIFTY', indices=True)
        
        if options_data is not None and not options_data.empty:
            logger.info(f"✅ Got {len(options_data)} options records")
            logger.info(f"📋 Columns: {list(options_data.columns)}")
            
            # Get spot price
            if 'Strike_Price' in options_data.columns:
                strikes = sorted(options_data['Strike_Price'].unique())
                current_price = strikes[len(strikes)//2]  # Middle strike as approximation
                logger.info(f"💰 Estimated spot price: ₹{current_price:,.0f}")
                
                # Find ATM strike
                atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                logger.info(f"🎯 ATM Strike: ₹{atm_strike:,.0f}")
                
                # Basic OI analysis
                atm_data = options_data[options_data['Strike_Price'] == atm_strike]
                if not atm_data.empty:
                    call_oi = float(atm_data['CALLS_OI'].iloc[0]) if 'CALLS_OI' in atm_data.columns else 0
                    put_oi = float(atm_data['PUTS_OI'].iloc[0]) if 'PUTS_OI' in atm_data.columns else 0
                    call_change = float(atm_data['CALLS_Chng_in_OI'].iloc[0]) if 'CALLS_Chng_in_OI' in atm_data.columns else 0
                    put_change = float(atm_data['PUTS_Chng_in_OI'].iloc[0]) if 'PUTS_Chng_in_OI' in atm_data.columns else 0
                    
                    pcr = put_oi / call_oi if call_oi > 0 else 0
                    
                    logger.info(f"📈 ATM OI Analysis:")
                    logger.info(f"   Call OI: {call_oi:,.0f}")
                    logger.info(f"   Put OI: {put_oi:,.0f}")
                    logger.info(f"   PCR: {pcr:.2f}")
                    logger.info(f"   Call OI Change: {call_change:,.0f}")
                    logger.info(f"   Put OI Change: {put_change:,.0f}")
                    
                    # Generate signal
                    signal = "NEUTRAL"
                    if pcr > 0.9 and put_change > 0 and call_change < 0:
                        signal = "BULLISH"
                    elif pcr < 0.8 and call_change > 0 and put_change < 0:
                        signal = "BEARISH"
                    elif 0.8 <= pcr <= 1.2 and call_change > 0 and put_change > 0:
                        signal = "RANGE"
                    
                    logger.info(f"🎯 Signal: {signal}")
                    
                    # Create simple record
                    record = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'index': 'NIFTY',
                        'atm_strike': atm_strike,
                        'spot_price': current_price,
                        'signal_type': signal,
                        'pcr': pcr,
                        'atm_call_oi': call_oi,
                        'atm_put_oi': put_oi,
                        'atm_call_oi_change': call_change,
                        'atm_put_oi_change': put_change
                    }
                    
                    logger.info(f"📊 Record created: {record}")
                    
                else:
                    logger.warning("⚠️ No ATM data found")
            else:
                logger.error("❌ No Strike_Price column found")
        else:
            logger.error("❌ No options data retrieved")
        
        logger.info("✅ Basic options test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_options()
