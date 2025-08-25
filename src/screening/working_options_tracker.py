#!/usr/bin/env python3
"""
Working Options Tracker - ATM ± 2 Strikes OI-Based Strategy
Simple and working options tracker for NIFTY and BANKNIFTY.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import os
import schedule
import time
import threading

from src.nsedata.NseUtility import NseUtils


class WorkingOptionsTracker:
    """Working Options Tracker for ATM ± 2 strikes OI-based strategy."""
    
    def __init__(self):
        """Initialize Working Options Tracker."""
        self.logger = logging.getLogger(__name__)
        self.nse = NseUtils()
        self.results_dir = "results/options_tracker"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # CSV file for tracking
        self.tracking_file = os.path.join(self.results_dir, "options_tracking.csv")
        self._initialize_tracking_file()
    
    def _initialize_tracking_file(self):
        """Initialize the tracking CSV file."""
        if not os.path.exists(self.tracking_file):
            columns = [
                'timestamp', 'index', 'atm_strike', 'spot_price', 'signal_type',
                'confidence', 'pcr', 'atm_call_oi', 'atm_put_oi', 'atm_call_oi_change',
                'atm_put_oi_change', 'support_level', 'resistance_level', 'suggested_trade',
                'entry_price', 'stop_loss', 'target', 'iv_regime', 'expiry_date',
                'updated_premium', 'updated_spot_price', 'result_status'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.tracking_file, index=False)
            self.logger.info(f"📊 Initialized tracking file: {self.tracking_file}")
    
    def run_analysis(self, index: str = 'NIFTY') -> Dict[str, Any]:
        """Run options analysis and save to tracking file."""
        self.logger.info(f"🎯 Running {index} options analysis...")
        
        try:
            # 1. Get options chain
            options_data = self._get_options_chain(index)
            if options_data is None or options_data.empty:
                return {'error': f'Could not get options data for {index}'}
            
            # 2. Get current spot price
            current_price = self._get_spot_price(options_data)
            if not current_price:
                return {'error': f'Could not get spot price for {index}'}
            
            # 3. Find ATM strike
            atm_strike = self._find_atm_strike(current_price, options_data)
            if not atm_strike:
                return {'error': f'Could not find ATM strike for {index}'}
            
            # 4. Analyze OI patterns at ATM ± 2 strikes
            oi_analysis = self._analyze_oi_patterns(options_data, atm_strike, current_price)
            
            # 5. Generate signal based on strategy rules
            signal = self._generate_signal(oi_analysis, current_price)
            
            # 6. Create tracking record
            record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'index': index,
                'atm_strike': atm_strike,
                'spot_price': current_price,
                'signal_type': signal['signal_type'],
                'confidence': signal['confidence'],
                'pcr': oi_analysis['pcr'],
                'atm_call_oi': oi_analysis['atm_call_oi'],
                'atm_put_oi': oi_analysis['atm_put_oi'],
                'atm_call_oi_change': oi_analysis['atm_call_oi_change'],
                'atm_put_oi_change': oi_analysis['atm_put_oi_change'],
                'support_level': signal['support'],
                'resistance_level': signal['resistance'],
                'suggested_trade': signal['suggested_trade'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'target': signal['target'],
                'iv_regime': oi_analysis['iv_regime'],
                'expiry_date': self._get_next_expiry(),
                'updated_premium': '',
                'updated_spot_price': '',
                'result_status': 'PENDING'
            }
            
            # 7. Save to CSV
            self._save_to_tracking_file(record)
            
            # 8. Update previous pending records
            self._update_previous_records(index)
            
            self.logger.info(f"✅ {index} analysis completed - Signal: {signal['signal_type']}")
            return record
            
        except Exception as e:
            self.logger.error(f"❌ Error in {index} analysis: {e}")
            return {'error': str(e)}
    
    def _get_options_chain(self, index: str) -> Optional[pd.DataFrame]:
        """Get options chain data using NseUtility."""
        try:
            # Try live options chain first (more real-time)
            options_data = self.nse.get_live_option_chain(index, indices=True)
            
            if options_data is None or options_data.empty:
                # Fallback to regular options chain
                options_data = self.nse.get_option_chain(index, indices=True)
            
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error getting options chain for {index}: {e}")
            return None
    
    def _get_spot_price(self, options_data: pd.DataFrame) -> Optional[float]:
        """Get current spot price from options data."""
        try:
            # Try to get underlyingValue from options data
            if 'underlyingValue' in options_data.columns:
                underlying_values = options_data['underlyingValue'].dropna()
                if not underlying_values.empty:
                    return float(underlying_values.iloc[0])
            
            # If no underlyingValue, estimate from middle strike
            if 'Strike_Price' in options_data.columns:
                strikes = options_data['Strike_Price'].unique()
                strikes = sorted(strikes)
                return float(strikes[len(strikes)//2])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting spot price: {e}")
            return None
    
    def _find_atm_strike(self, current_price: float, options_data: pd.DataFrame) -> Optional[float]:
        """Find ATM strike closest to current price."""
        try:
            strike_col = 'Strike_Price' if 'Strike_Price' in options_data.columns else 'strikePrice'
            
            if strike_col not in options_data.columns:
                self.logger.error(f"No strike price column found")
                return None
            
            strikes = options_data[strike_col].unique()
            strikes = sorted(strikes)
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            return float(atm_strike)
            
        except Exception as e:
            self.logger.error(f"Error finding ATM strike: {e}")
            return None
    
    def _analyze_oi_patterns(self, options_data: pd.DataFrame, atm_strike: float, current_price: float) -> Dict[str, Any]:
        """Analyze OI patterns at ATM ± 2 strikes."""
        try:
            # Get ATM ± 2 strikes
            strikes = sorted(options_data['Strike_Price'].unique())
            atm_index = strikes.index(atm_strike)
            
            start_idx = max(0, atm_index - 2)
            end_idx = min(len(strikes), atm_index + 3)
            strikes_to_analyze = strikes[start_idx:end_idx]
            
            self.logger.info(f"Analyzing strikes: {strikes_to_analyze} (ATM: {atm_strike})")
            
            # Analyze OI data
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
            
            self.logger.info(f"OI Analysis - PCR: {pcr:.2f}, ATM Call OI: {atm_call_oi:,.0f}, ATM Put OI: {atm_put_oi:,.0f}")
            
            return {
                'pcr': pcr,
                'atm_call_oi': atm_call_oi,
                'atm_put_oi': atm_put_oi,
                'atm_call_oi_change': atm_call_oi_change,
                'atm_put_oi_change': atm_put_oi_change,
                'iv_regime': 'MEDIUM_VOLATILITY'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing OI patterns: {e}")
            return {'pcr': 0, 'atm_call_oi': 0, 'atm_put_oi': 0, 'atm_call_oi_change': 0, 'atm_put_oi_change': 0, 'iv_regime': 'UNKNOWN'}
    
    def _generate_signal(self, oi_analysis: Dict, current_price: float) -> Dict[str, Any]:
        """Generate trading signal based on strategy rules."""
        pcr = oi_analysis.get('pcr', 0)
        atm_call_oi_change = oi_analysis.get('atm_call_oi_change', 0)
        atm_put_oi_change = oi_analysis.get('atm_put_oi_change', 0)
        
        signal = {
            'signal_type': 'NEUTRAL',
            'confidence': 50.0,
            'suggested_trade': 'Wait for clearer signal',
            'entry_price': current_price,
            'stop_loss': current_price * 0.98,
            'target': current_price * 1.02,
            'support': current_price * 0.98,
            'resistance': current_price * 1.02
        }
        
        # Strategy Rules Implementation
        
        # 1. Bullish Signal Rules
        if (pcr > 0.9 and 
            atm_put_oi_change > 0 and  # Put OI ↑ at ATM/support
            atm_call_oi_change < 0):   # Call OI ↓ (unwinding)
            
            signal.update({
                'signal_type': 'BULLISH',
                'confidence': min(90, 60 + (pcr - 0.9) * 100),
                'suggested_trade': 'Buy Call (ATM/ITM) or Bull Call Spread',
                'target': current_price * 1.03
            })
        
        # 2. Bearish Signal Rules
        elif (pcr < 0.8 and 
              atm_call_oi_change > 0 and  # Call OI ↑ at ATM/resistance
              atm_put_oi_change < 0):     # Put OI ↓ (unwinding)
            
            signal.update({
                'signal_type': 'BEARISH',
                'confidence': min(90, 60 + (0.8 - pcr) * 100),
                'suggested_trade': 'Buy Put (ATM/ITM) or Bear Put Spread',
                'stop_loss': current_price * 1.02,
                'target': current_price * 0.97
            })
        
        # 3. Range-Bound Signal Rules
        elif (0.8 <= pcr <= 1.2 and 
              atm_call_oi_change > 0 and  # Both Call & Put OI ↑ near ATM
              atm_put_oi_change > 0):
            
            signal.update({
                'signal_type': 'RANGE',
                'confidence': 70.0,
                'suggested_trade': 'Sell Straddle/Strangle',
                'stop_loss': current_price * 1.015,
                'target': current_price * 0.985
            })
        
        self.logger.info(f"Signal generated: {signal['signal_type']} (Confidence: {signal['confidence']:.1f}%)")
        return signal
    
    def _get_next_expiry(self) -> str:
        """Get next expiry date (simplified)."""
        return (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    
    def _save_to_tracking_file(self, record: Dict[str, Any]):
        """Save record to tracking CSV file."""
        try:
            df = pd.DataFrame([record])
            df.to_csv(self.tracking_file, mode='a', header=False, index=False)
            self.logger.info(f"📊 Record saved to tracking file")
        except Exception as e:
            self.logger.error(f"Error saving to tracking file: {e}")
    
    def _update_previous_records(self, index: str):
        """Update previous pending records with current data."""
        try:
            if not os.path.exists(self.tracking_file):
                return
                
            df = pd.read_csv(self.tracking_file)
            
            # Get current price
            options_data = self._get_options_chain(index)
            if options_data is None or options_data.empty:
                return
                
            current_price = self._get_spot_price(options_data)
            if not current_price:
                return
            
            # Update pending records for this index
            mask = (df['index'] == index) & (df['result_status'] == 'PENDING')
            if mask.any():
                df.loc[mask, 'updated_spot_price'] = current_price
                df.loc[mask, 'result_status'] = 'UPDATED'
                df.to_csv(self.tracking_file, index=False)
                self.logger.info(f"📊 Updated {mask.sum()} previous records for {index}")
                
        except Exception as e:
            self.logger.error(f"Error updating previous records: {e}")
    
    def start_scheduler(self, interval_minutes: int = 15):
        """Start the scheduler to run analysis every 15 minutes."""
        self.logger.info(f"🚀 Starting options tracker scheduler (every {interval_minutes} minutes)")
        
        # Schedule for both NIFTY and BANKNIFTY
        schedule.every(interval_minutes).minutes.do(self.run_analysis, 'NIFTY')
        schedule.every(interval_minutes).minutes.do(self.run_analysis, 'BANKNIFTY')
        
        # Run in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        self.logger.info("✅ Options tracker scheduler started successfully")
    
    def get_latest_analysis(self, index: str = 'NIFTY') -> Optional[Dict[str, Any]]:
        """Get the latest analysis result."""
        try:
            if not os.path.exists(self.tracking_file):
                return None
                
            df = pd.read_csv(self.tracking_file)
            if df.empty:
                return None
            
            latest = df[df['index'] == index].iloc[-1] if index in df['index'].values else None
            return latest.to_dict() if latest is not None else None
            
        except Exception as e:
            self.logger.error(f"Error getting latest analysis: {e}")
            return None
    
    def get_analysis_history(self, index: str = 'NIFTY', days: int = 7) -> pd.DataFrame:
        """Get analysis history for the specified period."""
        try:
            if not os.path.exists(self.tracking_file):
                return pd.DataFrame()
                
            df = pd.read_csv(self.tracking_file)
            if df.empty:
                return pd.DataFrame()
            
            # Filter by index and date
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            
            filtered_df = df[
                (df['index'] == index) & 
                (df['timestamp'] >= cutoff_date)
            ].sort_values('timestamp', ascending=False)
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error getting analysis history: {e}")
            return pd.DataFrame()


# Global instance
working_options_tracker = WorkingOptionsTracker()
