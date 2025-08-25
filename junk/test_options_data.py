#!/usr/bin/env python3
"""
Test Options Data Access
Verify that we can access NIFTY and BANKNIFTY options data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from loguru import logger

from src.nsedata.NseUtility import NseUtils
from src.data.database.duckdb_manager import DatabaseManager

def test_options_data_access():
    """Test options data access for NIFTY and BANKNIFTY."""
    logger.info("🧪 Testing Options Data Access")
    logger.info("=" * 50)
    
    try:
        # Initialize NSE utility
        nse = NseUtils()
        db_manager = DatabaseManager()
        
        # Test indices
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            logger.info(f"\n📊 Testing {index}...")
            
            # 1. Get current price
            logger.info(f"1️⃣ Getting current {index} price...")
            
            # Try database first
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            df = db_manager.get_price_data(index, start_date, end_date)
            if not df.empty:
                current_price = float(df['close_price'].iloc[-1])
                logger.info(f"   ✅ Database price: ₹{current_price:.2f}")
            else:
                logger.warning(f"   ⚠️ No database data for {index}")
                current_price = None
            
            # Try NSE API as fallback
            if current_price is None:
                try:
                    price_info = nse.price_info(index)
                    if price_info and 'LastTradedPrice' in price_info:
                        current_price = float(price_info['LastTradedPrice'])
                        logger.info(f"   ✅ NSE API price: ₹{current_price:.2f}")
                    else:
                        logger.error(f"   ❌ No NSE price data for {index}")
                        continue
                except Exception as e:
                    logger.error(f"   ❌ NSE API error: {e}")
                    continue
            
            # 2. Get options chain
            logger.info(f"2️⃣ Getting {index} options chain...")
            try:
                options_data = nse.get_option_chain(index, indices=True)
                
                if options_data is not None and not options_data.empty:
                    logger.info(f"   ✅ Options data retrieved successfully")
                    logger.info(f"   📊 Total records: {len(options_data)}")
                    logger.info(f"   📋 Columns: {list(options_data.columns)}")
                    
                    # Show sample data
                    logger.info(f"   📋 Sample data:")
                    logger.info(f"      {options_data.head(3).to_string()}")
                    
                    # Find ATM strike
                    if 'strikePrice' in options_data.columns:
                        strikes = options_data['strikePrice'].unique()
                        strikes = sorted(strikes)
                        atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                        logger.info(f"   🎯 ATM Strike: {atm_strike} (Current: {current_price})")
                        
                        # Analyze ATM ± 2 strikes
                        atm_index = strikes.index(atm_strike)
                        start_idx = max(0, atm_index - 2)
                        end_idx = min(len(strikes), atm_index + 3)
                        strikes_to_analyze = strikes[start_idx:end_idx]
                        
                        logger.info(f"   📊 Analyzing strikes: {strikes_to_analyze}")
                        
                        # Basic OI analysis
                        total_call_oi = 0
                        total_put_oi = 0
                        atm_call_oi = 0
                        atm_put_oi = 0
                        
                        for strike in strikes_to_analyze:
                            strike_data = options_data[options_data['strikePrice'] == strike]
                            
                            # Call data
                            call_data = strike_data[strike_data['instrumentType'] == 'CE']
                            if not call_data.empty and 'openInterest' in call_data.columns:
                                call_oi = int(call_data['openInterest'].iloc[0])
                                total_call_oi += call_oi
                                if strike == atm_strike:
                                    atm_call_oi = call_oi
                            
                            # Put data
                            put_data = strike_data[strike_data['instrumentType'] == 'PE']
                            if not put_data.empty and 'openInterest' in put_data.columns:
                                put_oi = int(put_data['openInterest'].iloc[0])
                                total_put_oi += put_oi
                                if strike == atm_strike:
                                    atm_put_oi = put_oi
                        
                        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                        
                        logger.info(f"   📈 OI Analysis:")
                        logger.info(f"      Total Call OI: {total_call_oi:,}")
                        logger.info(f"      Total Put OI: {total_put_oi:,}")
                        logger.info(f"      PCR: {pcr:.2f}")
                        logger.info(f"      ATM Call OI: {atm_call_oi:,}")
                        logger.info(f"      ATM Put OI: {atm_put_oi:,}")
                        
                        # Basic signal generation
                        signal = "NEUTRAL"
                        if pcr > 0.9:
                            signal = "BULLISH"
                        elif pcr < 0.8:
                            signal = "BEARISH"
                        
                        logger.info(f"   🎯 Signal: {signal}")
                        
                    else:
                        logger.warning(f"   ⚠️ No strike price column found")
                        
                else:
                    logger.error(f"   ❌ No options data for {index}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error getting options data: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n✅ Options data access test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from datetime import timedelta
    test_options_data_access()
