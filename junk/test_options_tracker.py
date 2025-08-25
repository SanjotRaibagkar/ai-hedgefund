#!/usr/bin/env python3
"""
Test Options Tracker
Test the options tracker functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def test_options_tracker():
    """Test options tracker functionality."""
    logger.info("🧪 Testing Options Tracker")
    logger.info("=" * 40)
    
    try:
        nse = NseUtils()
        
        # Test NIFTY
        logger.info("📊 Testing NIFTY options tracker...")
        
        # Get options data
        options_data = nse.get_live_option_chain('NIFTY', indices=True)
        
        if options_data is not None and not options_data.empty:
            logger.info(f"✅ Got {len(options_data)} options records")
            
            # Get spot price
            current_price = None
            if 'Strike_Price' in options_data.columns:
                strikes = sorted(options_data['Strike_Price'].unique())
                current_price = float(strikes[len(strikes)//2])
                logger.info(f"💰 Spot price: ₹{current_price:,.0f}")
                
                # Find ATM strike
                atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                logger.info(f"🎯 ATM Strike: ₹{atm_strike:,.0f}")
                
                # Analyze ATM ± 2 strikes
                atm_index = strikes.index(atm_strike)
                start_idx = max(0, atm_index - 2)
                end_idx = min(len(strikes), atm_index + 3)
                strikes_to_analyze = strikes[start_idx:end_idx]
                
                logger.info(f"📊 Analyzing strikes: {strikes_to_analyze}")
                
                # OI analysis
                total_call_oi = 0
                total_put_oi = 0
                atm_call_oi = 0
                atm_put_oi = 0
                atm_call_oi_change = 0
                atm_put_oi_change = 0
                
                for strike in strikes_to_analyze:
                    strike_data = options_data[options_data['Strike_Price'] == strike]
                    
                    if not strike_data.empty:
                        call_oi = float(strike_data['CALLS_OI'].iloc[0]) if 'CALLS_OI' in strike_data.columns else 0
                        put_oi = float(strike_data['PUTS_OI'].iloc[0]) if 'PUTS_OI' in strike_data.columns else 0
                        call_oi_change = float(strike_data['CALLS_Chng_in_OI'].iloc[0]) if 'CALLS_Chng_in_OI' in strike_data.columns else 0
                        put_oi_change = float(strike_data['PUTS_Chng_in_OI'].iloc[0]) if 'PUTS_Chng_in_OI' in strike_data.columns else 0
                        
                        total_call_oi += call_oi
                        total_put_oi += put_oi
                        
                        if strike == atm_strike:
                            atm_call_oi = call_oi
                            atm_put_oi = put_oi
                            atm_call_oi_change = call_oi_change
                            atm_put_oi_change = put_oi_change
                
                pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                
                logger.info(f"📈 OI Analysis:")
                logger.info(f"   PCR: {pcr:.2f}")
                logger.info(f"   ATM Call OI: {atm_call_oi:,.0f}")
                logger.info(f"   ATM Put OI: {atm_put_oi:,.0f}")
                logger.info(f"   ATM Call OI Change: {atm_call_oi_change:,.0f}")
                logger.info(f"   ATM Put OI Change: {atm_put_oi_change:,.0f}")
                
                # Generate signal
                signal = "NEUTRAL"
                confidence = 50.0
                suggested_trade = "Wait for clearer signal"
                
                # Strategy Rules
                if pcr > 0.9 and atm_put_oi_change > 0 and atm_call_oi_change < 0:
                    signal = "BULLISH"
                    confidence = min(90, 60 + (pcr - 0.9) * 100)
                    suggested_trade = "Buy Call (ATM/ITM) or Bull Call Spread"
                elif pcr < 0.8 and atm_call_oi_change > 0 and atm_put_oi_change < 0:
                    signal = "BEARISH"
                    confidence = min(90, 60 + (0.8 - pcr) * 100)
                    suggested_trade = "Buy Put (ATM/ITM) or Bear Put Spread"
                elif 0.8 <= pcr <= 1.2 and atm_call_oi_change > 0 and atm_put_oi_change > 0:
                    signal = "RANGE"
                    confidence = 70.0
                    suggested_trade = "Sell Straddle/Strangle"
                
                logger.info(f"🎯 Signal: {signal} (Confidence: {confidence:.1f}%)")
                logger.info(f"💡 Suggested Trade: {suggested_trade}")
                
                # Create tracking record
                record = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'index': 'NIFTY',
                    'atm_strike': atm_strike,
                    'spot_price': current_price,
                    'signal_type': signal,
                    'confidence': confidence,
                    'pcr': pcr,
                    'atm_call_oi': atm_call_oi,
                    'atm_put_oi': atm_put_oi,
                    'atm_call_oi_change': atm_call_oi_change,
                    'atm_put_oi_change': atm_put_oi_change,
                    'support_level': current_price * 0.98,
                    'resistance_level': current_price * 1.02,
                    'suggested_trade': suggested_trade,
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.98,
                    'target': current_price * 1.02,
                    'iv_regime': 'MEDIUM_VOLATILITY',
                    'expiry_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'updated_premium': '',
                    'updated_spot_price': '',
                    'result_status': 'PENDING'
                }
                
                logger.info(f"📊 Tracking record created successfully")
                logger.info(f"📋 Record keys: {list(record.keys())}")
                
                # Test BANKNIFTY too
                logger.info("\n📊 Testing BANKNIFTY options tracker...")
                
                banknifty_data = nse.get_live_option_chain('BANKNIFTY', indices=True)
                
                if banknifty_data is not None and not banknifty_data.empty:
                    logger.info(f"✅ Got {len(banknifty_data)} BANKNIFTY options records")
                    
                    # Quick analysis for BANKNIFTY
                    if 'Strike_Price' in banknifty_data.columns:
                        strikes = sorted(banknifty_data['Strike_Price'].unique())
                        current_price = float(strikes[len(strikes)//2])
                        atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                        
                        atm_data = banknifty_data[banknifty_data['Strike_Price'] == atm_strike]
                        if not atm_data.empty:
                            call_oi = float(atm_data['CALLS_OI'].iloc[0]) if 'CALLS_OI' in atm_data.columns else 0
                            put_oi = float(atm_data['PUTS_OI'].iloc[0]) if 'PUTS_OI' in atm_data.columns else 0
                            pcr = put_oi / call_oi if call_oi > 0 else 0
                            
                            logger.info(f"💰 BANKNIFTY Spot: ₹{current_price:,.0f}")
                            logger.info(f"🎯 BANKNIFTY ATM: ₹{atm_strike:,.0f}")
                            logger.info(f"📈 BANKNIFTY PCR: {pcr:.2f}")
                else:
                    logger.warning("⚠️ No BANKNIFTY data retrieved")
                
            else:
                logger.error("❌ No Strike_Price column found")
        else:
            logger.error("❌ No options data retrieved")
        
        logger.info("✅ Options tracker test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_options_tracker()
