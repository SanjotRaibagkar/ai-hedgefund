#!/usr/bin/env python3
"""
Fixed Options Tracker
Fixes CSV formatting and adds performance tracking.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import schedule
import pandas as pd
from datetime import datetime
from loguru import logger

from src.nsedata.NseUtility import NseUtils

def run_options_analysis(index: str = 'NIFTY'):
    """Run options analysis with proper CSV formatting."""
    logger.info(f"🎯 Running {index} options analysis at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        nse = NseUtils()
        options_data = nse.get_live_option_chain(index, indices=True)
        
        if options_data is not None and not options_data.empty:
            # Get spot price from futures data (more accurate)
            current_price = None
            try:
                futures_data = nse.futures_data(index, indices=True)
                if futures_data is not None and not futures_data.empty:
                    # Get the first row (current month contract) for spot price
                    current_price = float(futures_data['lastPrice'].iloc[0])
                    logger.info(f"✅ Using futures data for {index}: ₹{current_price:,.2f}")
                else:
                    logger.warning(f"⚠️ No futures data for {index}, will use options fallback")
            except Exception as e:
                logger.warning(f"⚠️ Could not get futures data for {index}: {e}")
            
            # Fallback to options data if futures fails
            if current_price is None:
                strikes = sorted(options_data['Strike_Price'].unique())
                current_price = float(strikes[len(strikes)//2])
                logger.warning(f"⚠️ Using options fallback for {index}: ₹{current_price:,.2f}")
            
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            
            # OI analysis
            atm_data = options_data[options_data['Strike_Price'] == atm_strike]
            if not atm_data.empty:
                call_oi = float(atm_data['CALLS_OI'].iloc[0]) if 'CALLS_OI' in atm_data.columns else 0
                put_oi = float(atm_data['PUTS_OI'].iloc[0]) if 'PUTS_OI' in atm_data.columns else 0
                call_oi_change = float(atm_data['CALLS_Chng_in_OI'].iloc[0]) if 'CALLS_Chng_in_OI' in atm_data.columns else 0
                put_oi_change = float(atm_data['PUTS_Chng_in_OI'].iloc[0]) if 'PUTS_Chng_in_OI' in atm_data.columns else 0
                
                pcr = put_oi / call_oi if call_oi > 0 else 0
                
                # Generate signal
                signal = "NEUTRAL"
                confidence = 50.0
                suggested_trade = "Wait for clearer signal"
                
                if pcr > 0.9 and put_oi_change > 0 and call_oi_change < 0:
                    signal = "BULLISH"
                    confidence = min(90, 60 + (pcr - 0.9) * 100)
                    suggested_trade = "Buy Call (ATM/ITM) or Bull Call Spread"
                elif pcr < 0.8 and call_oi_change > 0 and put_oi_change < 0:
                    signal = "BEARISH"
                    confidence = min(90, 60 + (0.8 - pcr) * 100)
                    suggested_trade = "Buy Put (ATM/ITM) or Bear Put Spread"
                elif 0.8 <= pcr <= 1.2 and call_oi_change > 0 and put_oi_change > 0:
                    signal = "RANGE"
                    confidence = 70.0
                    suggested_trade = "Sell Straddle/Strangle"
                
                # Create properly formatted record
                new_record = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'index': index,
                    'atm_strike': atm_strike,
                    'initial_spot_price': current_price,
                    'signal_type': signal,
                    'confidence': confidence,
                    'pcr': pcr,
                    'atm_call_oi': call_oi,
                    'atm_put_oi': put_oi,
                    'atm_call_oi_change': call_oi_change,
                    'atm_put_oi_change': put_oi_change,
                    'suggested_trade': suggested_trade,
                    'current_spot_price': current_price,
                    'current_option_premium': 0,
                    'price_change_percent': 0,
                    'performance_status': 'ACTIVE'
                }
                
                # Save to CSV with proper formatting
                csv_file = "results/options_tracker/options_tracking.csv"
                os.makedirs(os.path.dirname(csv_file), exist_ok=True)
                
                # Create or load existing CSV
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                else:
                    # Create new CSV with headers
                    df = pd.DataFrame(columns=list(new_record.keys()))
                
                # Add new record
                df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
                df.to_csv(csv_file, index=False)
                
                # Update previous records with current prices
                update_previous_records(df, index, current_price, csv_file)
                
                logger.info(f"📊 {index} Analysis Results:")
                logger.info(f"   Spot Price: ₹{current_price:,.0f}")
                logger.info(f"   ATM Strike: ₹{atm_strike:,.0f}")
                logger.info(f"   PCR: {pcr:.2f}")
                logger.info(f"   Signal: {signal} (Confidence: {confidence:.1f}%)")
                logger.info(f"   Trade: {suggested_trade}")
                logger.info(f"📊 Record saved to {csv_file}")
                
        else:
            logger.error(f"❌ No options data for {index}")
            
    except Exception as e:
        logger.error(f"❌ Error in {index} analysis: {e}")

def update_previous_records(df, index, current_price, csv_file):
    """Update previous records with current prices."""
    try:
        # Get active records for this index
        mask = (df['index'] == index) & (df['performance_status'] == 'ACTIVE')
        active_indices = df[mask].index
        
        for idx in active_indices:
            initial_price = df.loc[idx, 'initial_spot_price']
            price_change = ((current_price - initial_price) / initial_price) * 100
            
            # Update current prices
            df.loc[idx, 'current_spot_price'] = current_price
            df.loc[idx, 'price_change_percent'] = price_change
            
            # Calculate performance status
            signal = df.loc[idx, 'signal_type']
            performance_status = calculate_performance_status(signal, price_change)
            df.loc[idx, 'performance_status'] = performance_status
            
            logger.info(f"📈 Updated {index} record: {price_change:+.2f}% change, Status: {performance_status}")
        
        # Save updated CSV
        df.to_csv(csv_file, index=False)
        
    except Exception as e:
        logger.error(f"❌ Error updating previous records: {e}")

def calculate_performance_status(signal, price_change):
    """Calculate performance status."""
    if signal == "BULLISH":
        if price_change > 0.5:
            return "PROFIT"
        elif price_change < -0.5:
            return "LOSS"
        else:
            return "NEUTRAL"
    elif signal == "BEARISH":
        if price_change < -0.5:
            return "PROFIT"
        elif price_change > 0.5:
            return "LOSS"
        else:
            return "NEUTRAL"
    else:
        if abs(price_change) < 0.3:
            return "ACCURATE"
        else:
            return "INACCURATE"

def run_nifty_analysis():
    """Run NIFTY analysis."""
    run_options_analysis('NIFTY')

def run_banknifty_analysis():
    """Run BANKNIFTY analysis."""
    run_options_analysis('BANKNIFTY')

def start_scheduler():
    """Start the scheduler."""
    logger.info("🚀 Starting Fixed Options Scheduler (every 15 minutes)")
    
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
        time.sleep(60)

if __name__ == "__main__":
    start_scheduler()
