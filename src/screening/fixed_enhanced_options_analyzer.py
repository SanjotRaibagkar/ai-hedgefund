#!/usr/bin/env python3
"""
Fixed Enhanced Options Analyzer - Unified Options Strategy Module
Single source of truth for options analysis used by scheduler, UI, and ML modules.
Implements ATM ± 2 strikes OI-based strategy with proper spot price extraction.
Does NOT require database access - uses direct NSE API calls.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import csv

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.nsedata.NseUtility import NseUtils


class FixedEnhancedOptionsAnalyzer:
    """Unified Options Analyzer with ATM ± 2 strikes OI-based strategy."""
    
    def __init__(self):
        """Initialize Fixed Enhanced Options Analyzer."""
        self.logger = logging.getLogger(__name__)
        self.nse = NseUtils()
        self.results_dir = "results/options_tracker"
        os.makedirs(self.results_dir, exist_ok=True)
        self.csv_file = os.path.join(self.results_dir, "option_tracker.csv")  # Fixed filename
        
    def _get_current_index_price(self, index: str) -> Optional[float]:
        """Get current index price from futures data (most accurate method)."""
        try:
            # Use futures_data for spot price (most accurate for indices)
            futures_data = self.nse.futures_data(index, indices=True)
            if futures_data is not None and not futures_data.empty:
                # Get the first row (current month contract) for spot price
                current_price = float(futures_data['lastPrice'].iloc[0])
                self.logger.info(f"Using futures data for {index}: {current_price:,.2f}")
                return current_price
            else:
                self.logger.warning(f"No futures data for {index}, will use options fallback")
                
                # Fallback to options data
                options_data = self.nse.get_live_option_chain(index, indices=True)
                if options_data is not None and isinstance(options_data, pd.DataFrame) and not options_data.empty:
                    strikes = sorted(options_data['Strike_Price'].unique())
                    current_price = float(strikes[len(strikes)//2])
                    self.logger.warning(f"Using options fallback for {index}: {current_price:,.2f}")
                    return current_price
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {index}: {e}")
            return None
    
    def _get_options_chain(self, index: str) -> Optional[pd.DataFrame]:
        """Get options chain data from NSE."""
        try:
            options_df = self.nse.get_live_option_chain(index, indices=True)
            if options_df is not None and isinstance(options_df, pd.DataFrame) and not options_df.empty:
                return options_df
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting options chain for {index}: {e}")
            return None
    
    def _find_atm_strike(self, current_price: float, options_data: pd.DataFrame) -> Optional[float]:
        """Find the At-The-Money strike closest to current price."""
        try:
            # Get unique strikes
            strikes = options_data['Strike_Price'].unique()
            strikes = sorted(strikes)
            
            # Find closest strike to current price
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            return float(atm_strike)
            
        except Exception as e:
            self.logger.error(f"Error finding ATM strike: {e}")
            return None
    
    def _analyze_oi_patterns_atm_plus_minus_2(self, options_data: pd.DataFrame, atm_strike: float, current_price: float) -> Dict[str, Any]:
        """Analyze OI patterns at ATM ± 2 strikes."""
        try:
            # Define strike range (ATM ± 2 strikes)
            strike_range = 2
            strikes_to_analyze = []
            
            # Get all available strikes
            all_strikes = sorted(options_data['Strike_Price'].unique())
            atm_index = all_strikes.index(atm_strike)
            
            # Get ATM ± 2 strikes
            start_idx = max(0, atm_index - strike_range)
            end_idx = min(len(all_strikes), atm_index + strike_range + 1)
            strikes_to_analyze = all_strikes[start_idx:end_idx]
            
            analysis = {
                'atm_strike': atm_strike,
                'analyzed_strikes': strikes_to_analyze,
                'pcr': 0.0,
                'atm_call_oi': 0,
                'atm_put_oi': 0,
                'atm_call_oi_change': 0,
                'atm_put_oi_change': 0
            }
            
            # Analyze each strike
            total_call_oi = 0
            total_put_oi = 0
            
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
                        analysis['atm_call_oi'] = call_oi
                        analysis['atm_put_oi'] = put_oi
                        analysis['atm_call_oi_change'] = call_oi_change
                        analysis['atm_put_oi_change'] = put_oi_change
            
            analysis['pcr'] = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing OI patterns: {e}")
            return {}
    
    def _generate_signal(self, oi_analysis: Dict[str, Any], current_price: float, atm_strike: float, index: str) -> Dict[str, Any]:
        """Generate trading signal based on strategy rules."""
        try:
            pcr = oi_analysis.get('pcr', 0)
            atm_call_oi_change = oi_analysis.get('atm_call_oi_change', 0)
            atm_put_oi_change = oi_analysis.get('atm_put_oi_change', 0)
            
            # Initialize signal
            signal = "NEUTRAL"
            confidence = 50.0
            suggested_trade = "Wait for clearer signal"
            
            # Strategy Rules (ATM ± 2 strikes OI based)
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
            
            return {
                'signal': signal,
                'confidence': confidence,
                'suggested_trade': suggested_trade,
                'pcr': pcr,
                'atm_call_oi_change': atm_call_oi_change,
                'atm_put_oi_change': atm_put_oi_change
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 50.0, 'suggested_trade': 'Error in analysis'}
    
    def _calculate_support_resistance(self, oi_analysis: Dict[str, Any], current_price: float) -> Dict[str, float]:
        """Calculate support and resistance levels based on OI."""
        try:
            # For now, return current price as both support and resistance
            # This can be enhanced later with actual OI-based calculation
            return {
                'support': current_price,
                'resistance': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {'support': current_price, 'resistance': current_price}
    
    def analyze_index_options(self, index: str = 'NIFTY') -> Dict[str, Any]:
        """
        Analyze options for Nifty or BankNifty using ATM ± 2 strikes strategy.
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            
        Returns:
            Dictionary with options analysis results
        """
        self.logger.info(f"Analyzing {index} options with ATM ± 2 strikes strategy")
        
        try:
            # Get current index price from futures data (most accurate)
            current_price = self._get_current_index_price(index)
            if not current_price:
                self.logger.error(f"Could not fetch {index} current price")
                return {}
            
            # Get options chain
            options_data = self._get_options_chain(index)
            if options_data is None or not isinstance(options_data, pd.DataFrame) or options_data.empty:
                self.logger.error(f"Could not fetch {index} options data")
                return {}
            
            # Find ATM strike and analyze ± 2 strikes
            atm_strike = self._find_atm_strike(current_price, options_data)
            if atm_strike is None:
                self.logger.error(f"Could not find ATM strike for {index}")
                return {}
            
            # Analyze OI patterns at ATM ± 2 strikes
            oi_analysis = self._analyze_oi_patterns_atm_plus_minus_2(options_data, atm_strike, current_price)
            if not oi_analysis:
                self.logger.error(f"Could not analyze OI patterns for {index}")
                return {}
            
            # Generate signal based on strategy rules
            signal = self._generate_signal(oi_analysis, current_price, atm_strike, index)
            
            # Calculate support and resistance levels
            support_resistance = self._calculate_support_resistance(oi_analysis, current_price)
            
            # Create comprehensive result
            result = {
                'index': index,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'atm_strike': atm_strike,
                'signal': signal,
                'oi_analysis': oi_analysis,
                'support_resistance': support_resistance
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {index} options: {e}")
            return {}
    
    def save_to_csv(self, result: Dict[str, Any]) -> bool:
        """Save analysis result to CSV with proper formatting."""
        try:
            # Prepare data for CSV (matching the expected format)
            csv_data = {
                'timestamp': result['timestamp'],
                'index': result['index'],
                'atm_strike': result['atm_strike'],
                'initial_spot_price': result['current_price'],
                'current_spot_price': result['current_price'],  # Will be updated in next run
                'strike_premium_price': 0,  # Placeholder for option premium
                'current_option_premium': 0,  # Will be updated in next run
                'pcr': result['oi_analysis']['pcr'],
                'signal': result['signal']['signal'],
                'confidence': result['signal']['confidence'],
                'suggested_trade': result['signal']['suggested_trade'],
                'strongest_support': result['support_resistance']['support'],
                'strongest_resistance': result['support_resistance']['resistance'],
                'atm_call_oi': result['oi_analysis']['atm_call_oi'],
                'atm_put_oi': result['oi_analysis']['atm_put_oi'],
                'atm_call_oi_change': result['oi_analysis']['atm_call_oi_change'],
                'atm_put_oi_change': result['oi_analysis']['atm_put_oi_change'],
                'price_change_percent': 0.0,  # Will be calculated in next run
                'performance_status': 'ACTIVE'  # Will be updated in next run
            }
            
            # Check if CSV exists and has headers
            file_exists = os.path.exists(self.csv_file)
            
            # Write to CSV
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_data)
            
            self.logger.info(f"Record saved to {self.csv_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
            return False
    
    def update_previous_records(self, index: str) -> bool:
        """Update previous records with current prices and performance."""
        try:
            if not os.path.exists(self.csv_file):
                return True
            
            # Read existing CSV
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return True
            
            # Get current price
            current_price = self._get_current_index_price(index)
            if not current_price:
                return False
            
            # Update active records for this index
            mask = (df['index'] == index) & (df['performance_status'] == 'ACTIVE')
            if mask.any():
                # Calculate price change
                initial_price = df.loc[mask, 'initial_spot_price'].iloc[0]
                price_change_percent = ((current_price - initial_price) / initial_price) * 100
                
                # Update current prices
                df.loc[mask, 'current_spot_price'] = current_price
                df.loc[mask, 'price_change_percent'] = price_change_percent
                
                # Update performance status based on signal
                signal = df.loc[mask, 'signal'].iloc[0]
                performance_status = self._calculate_performance_status(signal, price_change_percent)
                df.loc[mask, 'performance_status'] = performance_status
                
                # Save updated CSV
                df.to_csv(self.csv_file, index=False)
                
                self.logger.info(f"Updated {index} record: {price_change_percent:+.2f}% change, Status: {performance_status}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating previous records: {e}")
            return False
    
    def _calculate_performance_status(self, signal: str, price_change_percent: float) -> str:
        """Calculate performance status based on signal and price change."""
        if signal == "BULLISH":
            if price_change_percent > 0.5:
                return "ACCURATE"
            elif price_change_percent < -0.5:
                return "INACCURATE"
            else:
                return "NEUTRAL"
        elif signal == "BEARISH":
            if price_change_percent < -0.5:
                return "ACCURATE"
            elif price_change_percent > 0.5:
                return "INACCURATE"
            else:
                return "NEUTRAL"
        else:
            if abs(price_change_percent) < 0.5:
                return "ACCURATE"
            else:
                return "NEUTRAL"
    
    def run_analysis_and_save(self, index: str = 'NIFTY') -> bool:
        """Run complete analysis and save to CSV with performance tracking."""
        try:
            # Update previous records first
            self.update_previous_records(index)
            
            # Run new analysis
            result = self.analyze_index_options(index)
            if not result:
                return False
            
            # Save to CSV
            return self.save_to_csv(result)
            
        except Exception as e:
            self.logger.error(f"Error in run_analysis_and_save: {e}")
            return False
    
    def get_latest_analysis(self, index: str = 'NIFTY') -> Dict[str, Any]:
        """Get latest analysis result for UI display."""
        try:
            result = self.analyze_index_options(index)
            if not result:
                return {}
            
            # Format for UI display
            return {
                'index': result['index'],
                'timestamp': result['timestamp'],
                'current_price': result['current_price'],
                'atm_strike': result['atm_strike'],
                'pcr': result['oi_analysis']['pcr'],
                'signal': result['signal']['signal'],
                'confidence': result['signal']['confidence'],
                'suggested_trade': result['signal']['suggested_trade'],
                'support': result['support_resistance']['support'],
                'resistance': result['support_resistance']['resistance'],
                'atm_call_oi': result['oi_analysis']['atm_call_oi'],
                'atm_put_oi': result['oi_analysis']['atm_put_oi'],
                'atm_call_oi_change': result['oi_analysis']['atm_call_oi_change'],
                'atm_put_oi_change': result['oi_analysis']['atm_put_oi_change']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting latest analysis: {e}")
            return {}


# Convenience functions for external use
def run_options_analysis(index: str = 'NIFTY') -> Dict[str, Any]:
    """Convenience function to run options analysis."""
    analyzer = FixedEnhancedOptionsAnalyzer()
    return analyzer.analyze_index_options(index)


def run_analysis_and_save(index: str = 'NIFTY') -> bool:
    """Convenience function to run analysis and save to CSV."""
    analyzer = FixedEnhancedOptionsAnalyzer()
    return analyzer.run_analysis_and_save(index)


def get_latest_analysis(index: str = 'NIFTY') -> Dict[str, Any]:
    """Convenience function to get latest analysis for UI."""
    analyzer = FixedEnhancedOptionsAnalyzer()
    return analyzer.get_latest_analysis(index)


if __name__ == "__main__":
    # Test the analyzer
    import logging
    logging.basicConfig(level=logging.INFO)
    
    analyzer = FixedEnhancedOptionsAnalyzer()
    
    # Test both indices
    for index in ['NIFTY', 'BANKNIFTY']:
        print(f"\n{'='*50}")
        print(f"Testing {index}")
        print(f"{'='*50}")
        
        # Run analysis and save
        success = analyzer.run_analysis_and_save(index)
        print(f"Analysis and save for {index}: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Get latest analysis
        result = analyzer.get_latest_analysis(index)
        if result:
            print(f"Current Price: ₹{result['current_price']:,.2f}")
            print(f"ATM Strike: ₹{result['atm_strike']:,.2f}")
            print(f"PCR: {result['pcr']:.2f}")
            print(f"Signal: {result['signal']} ({result['confidence']:.1f}% confidence)")
            print(f"Trade: {result['suggested_trade']}")
        else:
            print(f"❌ No analysis result for {index}")
    
    print(f"\n{'='*50}")
    print("CSV file created at: results/options_tracker/option_tracker.csv")
    print(f"{'='*50}")
