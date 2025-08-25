#!/usr/bin/env python3
"""
Test Simple Options Tracker
Verify the options tracker functionality with NSE API.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def test_simple_options_tracker():
    """Test simple options tracker functionality."""
    logger.info("🧪 Testing Simple Options Tracker")
    logger.info("=" * 50)
    
    try:
        # Initialize NSE utility
        nse = NseUtils()
        
        # Test indices
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            logger.info(f"\n📊 Testing {index}...")
            
            # 1. Get options chain
            logger.info(f"1️⃣ Getting options chain for {index}...")
            try:
                # Try live options chain first
                options_data = nse.get_live_option_chain(index, indices=True)
                
                if options_data is None or options_data.empty:
                    # Fallback to regular options chain
                    options_data = nse.get_option_chain(index, indices=True)
                
                if options_data is not None and not options_data.empty:
                    logger.info(f"   ✅ Options data retrieved successfully")
                    logger.info(f"   📊 Total records: {len(options_data)}")
                    logger.info(f"   📋 Columns: {list(options_data.columns)}")
                    
                    # 2. Get spot price
                    logger.info(f"2️⃣ Getting spot price for {index}...")
                    current_price = None
                    
                    if 'underlyingValue' in options_data.columns:
                        underlying_values = options_data['underlyingValue'].dropna()
                        if not underlying_values.empty:
                            current_price = float(underlying_values.iloc[0])
                            logger.info(f"   ✅ Spot price from underlyingValue: ₹{current_price:.2f}")
                    
                    if current_price is None and 'Strike_Price' in options_data.columns:
                        strikes = options_data['Strike_Price'].unique()
                        strikes = sorted(strikes)
                        current_price = float(strikes[len(strikes)//2])
                        logger.info(f"   ⚠️ Estimated spot price from middle strike: ₹{current_price:.2f}")
                    
                    if current_price:
                        # 3. Find ATM strike
                        logger.info(f"3️⃣ Finding ATM strike for {index}...")
                        strike_col = 'Strike_Price' if 'Strike_Price' in options_data.columns else 'strikePrice'
                        
                        if strike_col in options_data.columns:
                            strikes = sorted(options_data[strike_col].unique())
                            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                            logger.info(f"   ✅ ATM Strike: {atm_strike} (Current: {current_price})")
                            
                            # 4. Analyze OI patterns
                            logger.info(f"4️⃣ Analyzing OI patterns for {index}...")
                            
                            if 'Strike_Price' in options_data.columns:
                                # Live format analysis
                                atm_index = strikes.index(atm_strike)
                                start_idx = max(0, atm_index - 2)
                                end_idx = min(len(strikes), atm_index + 3)
                                strikes_to_analyze = strikes[start_idx:end_idx]
                                
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
                                
                                logger.info(f"   📈 OI Analysis:")
                                logger.info(f"      PCR: {pcr:.2f}")
                                logger.info(f"      ATM Call OI: {atm_call_oi:,.0f}")
                                logger.info(f"      ATM Put OI: {atm_put_oi:,.0f}")
                                logger.info(f"      ATM Call OI Change: {atm_call_oi_change:,.0f}")
                                logger.info(f"      ATM Put OI Change: {atm_put_oi_change:,.0f}")
                                
                                # 5. Generate signal
                                logger.info(f"5️⃣ Generating signal for {index}...")
                                
                                signal = "NEUTRAL"
                                confidence = 50.0
                                
                                # Bullish Signal Rules
                                if pcr > 0.9 and atm_put_oi_change > 0 and atm_call_oi_change < 0:
                                    signal = "BULLISH"
                                    confidence = min(90, 60 + (pcr - 0.9) * 100)
                                # Bearish Signal Rules
                                elif pcr < 0.8 and atm_call_oi_change > 0 and atm_put_oi_change < 0:
                                    signal = "BEARISH"
                                    confidence = min(90, 60 + (0.8 - pcr) * 100)
                                # Range-Bound Signal Rules
                                elif 0.8 <= pcr <= 1.2 and atm_call_oi_change > 0 and atm_put_oi_change > 0:
                                    signal = "RANGE"
                                    confidence = 70.0
                                
                                logger.info(f"   🎯 Signal: {signal} (Confidence: {confidence:.1f}%)")
                                
                                # 6. Create tracking record
                                record = {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'index': index,
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
                                    'suggested_trade': f"{signal} trade strategy",
                                    'entry_price': current_price,
                                    'stop_loss': current_price * 0.98,
                                    'target': current_price * 1.02,
                                    'iv_regime': 'MEDIUM_VOLATILITY',
                                    'expiry_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                                    'updated_premium': '',
                                    'updated_spot_price': '',
                                    'result_status': 'PENDING'
                                }
                                
                                logger.info(f"   📊 Record created successfully")
                                logger.info(f"   📋 Sample record keys: {list(record.keys())}")
                                
                            else:
                                logger.warning(f"   ⚠️ Regular options format not implemented in test")
                        else:
                            logger.error(f"   ❌ No strike price column found")
                    else:
                        logger.error(f"   ❌ Could not determine spot price")
                        
                else:
                    logger.error(f"   ❌ No options data for {index}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error in options analysis: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n✅ Simple options tracker test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from datetime import timedelta
    test_simple_options_tracker()
