#!/usr/bin/env python3
"""
Improved Options Tracker
Tracks options signals and updates previous rows with current prices for performance analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import schedule
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from src.nsedata.NseUtility import NseUtils

class ImprovedOptionsTracker:
    """Improved options tracker with performance monitoring."""
    
    def __init__(self):
        """Initialize the tracker."""
        self.nse = NseUtils()
        self.csv_file = "results/options_tracker/options_tracking.csv"
        self.performance_csv = "results/options_tracker/options_performance.csv"
        
        # Create directories
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        
        # Initialize CSV files with proper headers
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with proper headers."""
        # Main tracking CSV
        if not os.path.exists(self.csv_file):
            header = [
                "timestamp",
                "index", 
                "atm_strike",
                "initial_spot_price",
                "signal_type",
                "confidence",
                "pcr",
                "atm_call_oi",
                "atm_put_oi", 
                "atm_call_oi_change",
                "atm_put_oi_change",
                "suggested_trade",
                "current_spot_price",
                "current_option_premium",
                "price_change_percent",
                "performance_status"
            ]
            df = pd.DataFrame(columns=header)
            df.to_csv(self.csv_file, index=False)
            logger.info(f"✅ Created {self.csv_file} with proper headers")
        
        # Performance tracking CSV
        if not os.path.exists(self.performance_csv):
            perf_header = [
                "signal_timestamp",
                "index",
                "initial_spot_price", 
                "current_spot_price",
                "spot_change_percent",
                "initial_signal",
                "current_signal",
                "signal_accuracy",
                "profit_loss_points",
                "performance_rating"
            ]
            df = pd.DataFrame(columns=perf_header)
            df.to_csv(self.performance_csv, index=False)
            logger.info(f"✅ Created {self.performance_csv} with proper headers")
    
    def run_options_analysis(self, index: str = 'NIFTY'):
        """Run options analysis and track performance."""
        logger.info(f"🎯 Running {index} options analysis at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Get options data
            options_data = self.nse.get_live_option_chain(index, indices=True)
            
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
                
                # Create new record
                new_record = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'index': index,
                    'atm_strike': atm_strike,
                    'initial_spot_price': current_price,
                    'signal_type': signal,
                    'confidence': confidence,
                    'pcr': pcr,
                    'atm_call_oi': atm_call_oi,
                    'atm_put_oi': atm_put_oi,
                    'atm_call_oi_change': atm_call_oi_change,
                    'atm_put_oi_change': atm_put_oi_change,
                    'suggested_trade': suggested_trade,
                    'current_spot_price': current_price,  # Same as initial for new record
                    'current_option_premium': 0,  # Will be updated later
                    'price_change_percent': 0,  # Will be calculated later
                    'performance_status': 'ACTIVE'  # Will be updated later
                }
                
                # Add new record to CSV
                self._add_new_record(new_record)
                
                # Update previous records with current prices
                self._update_previous_records(index, current_price)
                
                # Log results
                logger.info(f"📊 {index} Analysis Results:")
                logger.info(f"   Spot Price: ₹{current_price:,.0f}")
                logger.info(f"   ATM Strike: ₹{atm_strike:,.0f}")
                logger.info(f"   PCR: {pcr:.2f}")
                logger.info(f"   Signal: {signal} (Confidence: {confidence:.1f}%)")
                logger.info(f"   Trade: {suggested_trade}")
                
            else:
                logger.error(f"❌ No options data for {index}")
                
        except Exception as e:
            logger.error(f"❌ Error in {index} analysis: {e}")
    
    def _add_new_record(self, record: dict):
        """Add new record to CSV."""
        try:
            df = pd.read_csv(self.csv_file)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df.to_csv(self.csv_file, index=False)
            logger.info(f"📊 New record added to {self.csv_file}")
        except Exception as e:
            logger.error(f"❌ Error adding new record: {e}")
    
    def _update_previous_records(self, index: str, current_price: float):
        """Update previous records with current prices and performance."""
        try:
            df = pd.read_csv(self.csv_file)
            
            # Get records for this index that are still active
            mask = (df['index'] == index) & (df['performance_status'] == 'ACTIVE')
            active_records = df[mask].copy()
            
            if not active_records.empty:
                for idx, record in active_records.iterrows():
                    initial_price = record['initial_spot_price']
                    price_change = ((current_price - initial_price) / initial_price) * 100
                    
                    # Update current prices and performance
                    df.loc[idx, 'current_spot_price'] = current_price
                    df.loc[idx, 'price_change_percent'] = price_change
                    
                    # Determine performance status based on signal and price movement
                    signal = record['signal_type']
                    performance_status = self._calculate_performance_status(signal, price_change)
                    df.loc[idx, 'performance_status'] = performance_status
                    
                    # Log performance update
                    logger.info(f"📈 {index} Performance Update:")
                    logger.info(f"   Initial Price: ₹{initial_price:,.0f}")
                    logger.info(f"   Current Price: ₹{current_price:,.0f}")
                    logger.info(f"   Change: {price_change:+.2f}%")
                    logger.info(f"   Signal: {signal} → Status: {performance_status}")
                
                # Save updated CSV
                df.to_csv(self.csv_file, index=False)
                logger.info(f"📊 Updated {len(active_records)} previous records")
                
        except Exception as e:
            logger.error(f"❌ Error updating previous records: {e}")
    
    def _calculate_performance_status(self, signal: str, price_change: float) -> str:
        """Calculate performance status based on signal and price change."""
        if signal == "BULLISH":
            if price_change > 0.5:  # 0.5% gain
                return "PROFIT"
            elif price_change < -0.5:  # 0.5% loss
                return "LOSS"
            else:
                return "NEUTRAL"
        elif signal == "BEARISH":
            if price_change < -0.5:  # 0.5% decline
                return "PROFIT"
            elif price_change > 0.5:  # 0.5% gain
                return "LOSS"
            else:
                return "NEUTRAL"
        else:  # NEUTRAL or RANGE
            if abs(price_change) < 0.3:  # Within 0.3%
                return "ACCURATE"
            else:
                return "INACCURATE"
    
    def generate_performance_report(self):
        """Generate performance report."""
        try:
            df = pd.read_csv(self.csv_file)
            
            if df.empty:
                logger.info("📊 No data available for performance report")
                return
            
            # Calculate performance metrics
            total_signals = len(df)
            profitable_signals = len(df[df['performance_status'] == 'PROFIT'])
            loss_signals = len(df[df['performance_status'] == 'LOSS'])
            accurate_signals = len(df[df['performance_status'] == 'ACCURATE'])
            
            success_rate = (profitable_signals + accurate_signals) / total_signals * 100 if total_signals > 0 else 0
            
            logger.info(f"📊 Performance Report:")
            logger.info(f"   Total Signals: {total_signals}")
            logger.info(f"   Profitable: {profitable_signals}")
            logger.info(f"   Loss: {loss_signals}")
            logger.info(f"   Accurate (Neutral): {accurate_signals}")
            logger.info(f"   Success Rate: {success_rate:.1f}%")
            
            # Save performance summary
            perf_summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_signals': total_signals,
                'profitable_signals': profitable_signals,
                'loss_signals': loss_signals,
                'accurate_signals': accurate_signals,
                'success_rate': success_rate
            }
            
            perf_df = pd.DataFrame([perf_summary])
            perf_df.to_csv(self.performance_csv, mode='a', header=not os.path.exists(self.performance_csv), index=False)
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")

def run_nifty_analysis():
    """Run NIFTY analysis."""
    tracker = ImprovedOptionsTracker()
    tracker.run_options_analysis('NIFTY')

def run_banknifty_analysis():
    """Run BANKNIFTY analysis."""
    tracker = ImprovedOptionsTracker()
    tracker.run_options_analysis('BANKNIFTY')

def generate_report():
    """Generate performance report."""
    tracker = ImprovedOptionsTracker()
    tracker.generate_performance_report()

def start_scheduler():
    """Start the scheduler."""
    logger.info("🚀 Starting Improved Options Scheduler (every 15 minutes)")
    
    # Schedule both indices
    schedule.every(15).minutes.do(run_nifty_analysis)
    schedule.every(15).minutes.do(run_banknifty_analysis)
    
    # Schedule performance report every hour
    schedule.every().hour.do(generate_report)
    
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
