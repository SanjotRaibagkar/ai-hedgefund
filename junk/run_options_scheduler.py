#!/usr/bin/env python3
"""
Run Options Scheduler
Simple script to run options analysis every 15 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import schedule
from datetime import datetime
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def run_options_analysis(index: str = 'NIFTY'):
    """Run options analysis for given index."""
    logger.info(f"🎯 Running {index} options analysis at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        nse = NseUtils()
        
        # Get options data
        options_data = nse.get_live_option_chain(index, indices=True)
        
        if options_data is not None and not options_data.empty:
            # Get spot price
            strikes = sorted(options_data['Strike_Price'].unique())
            current_price = float(strikes[len(strikes)//2])
            
            # Find ATM strike
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            
            # Analyze ATM ± 2 strikes
            atm_index = strikes.index(atm_strike)
            start_idx = max(0, atm_index - 2)
            end_idx = min(len(strikes), atm_index + 3)
            strikes_to_analyze = strikes[start_idx:end_idx]
            
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
            
            # Log results
            logger.info(f"📊 {index} Analysis Results:")
            logger.info(f"   Spot Price: ₹{current_price:,.0f}")
            logger.info(f"   ATM Strike: ₹{atm_strike:,.0f}")
            logger.info(f"   PCR: {pcr:.2f}")
            logger.info(f"   Signal: {signal} (Confidence: {confidence:.1f}%)")
            logger.info(f"   Trade: {suggested_trade}")
            
            # Save to CSV (simple format)
            record = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{index},{atm_strike},{current_price},{signal},{confidence:.1f},{pcr:.2f},{atm_call_oi:,.0f},{atm_put_oi:,.0f},{atm_call_oi_change:,.0f},{atm_put_oi_change:,.0f},{suggested_trade}\n"
            
            csv_file = "results/options_tracker/options_tracking.csv"
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            
            # Write header if file doesn't exist
            if not os.path.exists(csv_file):
                header = "timestamp,index,atm_strike,spot_price,signal_type,confidence,pcr,atm_call_oi,atm_put_oi,atm_call_oi_change,atm_put_oi_change,suggested_trade\n"
                with open(csv_file, 'w') as f:
                    f.write(header)
            
            # Append record
            with open(csv_file, 'a') as f:
                f.write(record)
            
            logger.info(f"📊 Record saved to {csv_file}")
            
        else:
            logger.error(f"❌ No options data for {index}")
            
    except Exception as e:
        logger.error(f"❌ Error in {index} analysis: {e}")

def run_nifty_analysis():
    """Run NIFTY analysis."""
    run_options_analysis('NIFTY')

def run_banknifty_analysis():
    """Run BANKNIFTY analysis."""
    run_options_analysis('BANKNIFTY')

def start_scheduler():
    """Start the scheduler."""
    logger.info("🚀 Starting Options Scheduler (every 15 minutes)")
    
    # Schedule both indices
    schedule.every(15).minutes.do(run_nifty_analysis)
    schedule.every(15).minutes.do(run_banknifty_analysis)
    
    # Run initial analysis
    logger.info("📊 Running initial analysis...")
    run_nifty_analysis()
    run_banknifty_analysis()
    
    # Start scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    start_scheduler()
