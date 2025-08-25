#!/usr/bin/env python3
"""
Enhanced Options Analyzer - ATM ± 2 Strikes OI-Based Strategy
Implements specific strategy rules for Nifty and BankNifty options analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
from dataclasses import dataclass

from src.nsedata.NseUtility import NseUtils
from src.data.database.duckdb_manager import DatabaseManager


@dataclass
class OptionsSignal:
    """Options signal data structure."""
    timestamp: str
    index: str
    atm_strike: float
    spot_price: float
    signal_type: str  # 'BULLISH', 'BEARISH', 'RANGE'
    confidence: float
    pcr: float
    atm_call_oi: int
    atm_put_oi: int
    atm_call_oi_change: int
    atm_put_oi_change: int
    support_level: float
    resistance_level: float
    suggested_trade: str
    entry_price: float
    stop_loss: float
    target: float
    iv_regime: str
    expiry_date: str


class EnhancedOptionsAnalyzer:
    """Enhanced Options Analyzer with ATM ± 2 strikes OI-based strategy."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        """Initialize Enhanced Options Analyzer."""
        self.logger = logging.getLogger(__name__)
        self.nse = NseUtils()
        self.db_manager = DatabaseManager(db_path)
        self.results_dir = "results/options_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def analyze_index_options(self, index: str = 'NIFTY') -> Dict[str, Any]:
        """
        Analyze options for Nifty or BankNifty using ATM ± 2 strikes strategy.
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            
        Returns:
            Dictionary with options analysis results
        """
        self.logger.info(f"🎯 Analyzing {index} options with ATM ± 2 strikes strategy")
        
        try:
            # Get current index price
            current_price = self._get_current_index_price(index)
            if not current_price:
                self.logger.error(f"Could not fetch {index} current price")
                return {}
            
            # Get options chain
            options_data = self._get_options_chain(index)
            if not options_data:
                self.logger.error(f"Could not fetch {index} options data")
                return {}
            
            # Find ATM strike and analyze ± 2 strikes
            atm_strike = self._find_atm_strike(current_price, options_data)
            if not atm_strike:
                self.logger.error(f"Could not find ATM strike for {index}")
                return {}
            
            # Analyze OI patterns at ATM ± 2 strikes
            oi_analysis = self._analyze_oi_patterns_atm_plus_minus_2(options_data, atm_strike, current_price)
            
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
                'support_resistance': support_resistance,
                'strategy_metrics': self._calculate_strategy_metrics(oi_analysis, signal)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {index} options: {e}")
            return {}
    
    def _get_current_index_price(self, index: str) -> Optional[float]:
        """Get current index price from database or NSE."""
        try:
            # Try database first
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            df = self.db_manager.get_price_data(index, start_date, end_date)
            if not df.empty:
                return float(df['close_price'].iloc[-1])
            
            # Fallback to NSE API
            price_info = self.nse.price_info(index)
            if price_info and 'LastTradedPrice' in price_info:
                return float(price_info['LastTradedPrice'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {index}: {e}")
            return None
    
    def _get_options_chain(self, index: str) -> Optional[pd.DataFrame]:
        """Get options chain data from NSE."""
        try:
            options_df = self.nse.get_option_chain(index, indices=True)
            if options_df is not None and not options_df.empty:
                return options_df
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting options chain for {index}: {e}")
            return None
    
    def _find_atm_strike(self, current_price: float, options_data: pd.DataFrame) -> Optional[float]:
        """Find the At-The-Money strike closest to current price."""
        try:
            # Get unique strikes
            strikes = options_data['strikePrice'].unique()
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
            all_strikes = sorted(options_data['strikePrice'].unique())
            atm_index = all_strikes.index(atm_strike)
            
            # Get ATM ± 2 strikes
            start_idx = max(0, atm_index - strike_range)
            end_idx = min(len(all_strikes), atm_index + strike_range + 1)
            strikes_to_analyze = all_strikes[start_idx:end_idx]
            
            analysis = {
                'atm_strike': atm_strike,
                'analyzed_strikes': strikes_to_analyze,
                'calls_data': {},
                'puts_data': {},
                'pcr': 0.0,
                'atm_call_oi': 0,
                'atm_put_oi': 0,
                'atm_call_oi_change': 0,
                'atm_put_oi_change': 0
            }
            
            # Analyze each strike
            for strike in strikes_to_analyze:
                strike_data = options_data[options_data['strikePrice'] == strike]
                
                # Get call data
                call_data = strike_data[strike_data['instrumentType'] == 'CE']
                if not call_data.empty:
                    analysis['calls_data'][strike] = {
                        'oi': int(call_data['openInterest'].iloc[0]) if 'openInterest' in call_data.columns else 0,
                        'oi_change': int(call_data['changeinOpenInterest'].iloc[0]) if 'changeinOpenInterest' in call_data.columns else 0,
                        'iv': float(call_data['impliedVolatility'].iloc[0]) if 'impliedVolatility' in call_data.columns else 0,
                        'premium': float(call_data['lastPrice'].iloc[0]) if 'lastPrice' in call_data.columns else 0
                    }
                
                # Get put data
                put_data = strike_data[strike_data['instrumentType'] == 'PE']
                if not put_data.empty:
                    analysis['puts_data'][strike] = {
                        'oi': int(put_data['openInterest'].iloc[0]) if 'openInterest' in put_data.columns else 0,
                        'oi_change': int(put_data['changeinOpenInterest'].iloc[0]) if 'changeinOpenInterest' in put_data.columns else 0,
                        'iv': float(put_data['impliedVolatility'].iloc[0]) if 'impliedVolatility' in put_data.columns else 0,
                        'premium': float(put_data['lastPrice'].iloc[0]) if 'lastPrice' in put_data.columns else 0
                    }
            
            # Calculate PCR and ATM OI
            total_call_oi = sum(data['oi'] for data in analysis['calls_data'].values())
            total_put_oi = sum(data['oi'] for data in analysis['puts_data'].values())
            
            analysis['pcr'] = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Get ATM specific data
            if atm_strike in analysis['calls_data']:
                analysis['atm_call_oi'] = analysis['calls_data'][atm_strike]['oi']
                analysis['atm_call_oi_change'] = analysis['calls_data'][atm_strike]['oi_change']
            
            if atm_strike in analysis['puts_data']:
                analysis['atm_put_oi'] = analysis['puts_data'][atm_strike]['oi']
                analysis['atm_put_oi_change'] = analysis['puts_data'][atm_strike]['oi_change']
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing OI patterns: {e}")
            return {}
    
    def _generate_signal(self, oi_analysis: Dict, current_price: float, atm_strike: float, index: str) -> Dict[str, Any]:
        """Generate trading signal based on strategy rules."""
        try:
            signal = {
                'signal_type': 'NEUTRAL',
                'confidence': 0.0,
                'reasoning': [],
                'suggested_trade': '',
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'target': 0.0
            }
            
            pcr = oi_analysis.get('pcr', 0)
            atm_call_oi_change = oi_analysis.get('atm_call_oi_change', 0)
            atm_put_oi_change = oi_analysis.get('atm_put_oi_change', 0)
            
            # Strategy Rules Implementation
            
            # 1. Bullish Signal Rules
            if (pcr > 0.9 and 
                atm_put_oi_change > 0 and  # Put OI ↑ at ATM/support
                atm_call_oi_change < 0):   # Call OI ↓ (unwinding)
                
                signal['signal_type'] = 'BULLISH'
                signal['confidence'] = min(90, 60 + (pcr - 0.9) * 100)
                signal['reasoning'] = [
                    f"PCR > 0.9 (Current: {pcr:.2f})",
                    f"Put OI increasing at ATM: +{atm_put_oi_change}",
                    f"Call OI decreasing (unwinding): {atm_call_oi_change}"
                ]
                signal['suggested_trade'] = "Buy Call (ATM/ITM) or Bull Call Spread"
                
                # Calculate levels
                signal['entry_price'] = current_price
                signal['stop_loss'] = current_price * 0.98  # 2% below current
                signal['target'] = current_price * 1.03     # 3% above current
            
            # 2. Bearish Signal Rules
            elif (pcr < 0.8 and 
                  atm_call_oi_change > 0 and  # Call OI ↑ at ATM/resistance
                  atm_put_oi_change < 0):     # Put OI ↓ (unwinding)
                
                signal['signal_type'] = 'BEARISH'
                signal['confidence'] = min(90, 60 + (0.8 - pcr) * 100)
                signal['reasoning'] = [
                    f"PCR < 0.8 (Current: {pcr:.2f})",
                    f"Call OI increasing at ATM: +{atm_call_oi_change}",
                    f"Put OI decreasing (unwinding): {atm_put_oi_change}"
                ]
                signal['suggested_trade'] = "Buy Put (ATM/ITM) or Bear Put Spread"
                
                # Calculate levels
                signal['entry_price'] = current_price
                signal['stop_loss'] = current_price * 1.02  # 2% above current
                signal['target'] = current_price * 0.97     # 3% below current
            
            # 3. Range-Bound Signal Rules
            elif (0.8 <= pcr <= 1.2 and
                  atm_call_oi_change > 0 and  # Both Call & Put OI ↑ near ATM
                  atm_put_oi_change > 0):
                
                signal['signal_type'] = 'RANGE'
                signal['confidence'] = 70.0
                signal['reasoning'] = [
                    f"PCR ~ 1.0 (Current: {pcr:.2f})",
                    f"Both Call & Put OI increasing near ATM",
                    f"Call OI: +{atm_call_oi_change}, Put OI: +{atm_put_oi_change}"
                ]
                signal['suggested_trade'] = "Sell Straddle/Strangle"
                
                # Calculate levels for range trading
                signal['entry_price'] = current_price
                signal['stop_loss'] = current_price * 1.015  # 1.5% above/below
                signal['target'] = current_price * 0.985     # 1.5% below/above
            
            # 4. Neutral/No Clear Signal
            else:
                signal['signal_type'] = 'NEUTRAL'
                signal['confidence'] = 50.0
                signal['reasoning'] = [
                    f"PCR: {pcr:.2f} (No clear bias)",
                    f"Call OI change: {atm_call_oi_change}",
                    f"Put OI change: {atm_put_oi_change}",
                    "No clear pattern matching strategy rules"
                ]
                signal['suggested_trade'] = "Wait for clearer signal"
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {'signal_type': 'ERROR', 'confidence': 0.0, 'reasoning': [f"Error: {str(e)}"]}
    
    def _calculate_support_resistance(self, oi_analysis: Dict, current_price: float) -> Dict[str, float]:
        """Calculate support and resistance levels based on OI."""
        try:
            # Find strikes with highest OI (support/resistance)
            max_call_oi_strike = 0
            max_put_oi_strike = 0
            max_call_oi = 0
            max_put_oi = 0
            
            for strike, data in oi_analysis.get('calls_data', {}).items():
                if data['oi'] > max_call_oi:
                    max_call_oi = data['oi']
                    max_call_oi_strike = strike
            
            for strike, data in oi_analysis.get('puts_data', {}).items():
                if data['oi'] > max_put_oi:
                    max_put_oi = data['oi']
                    max_put_oi_strike = strike
            
            # Use OI-based levels or fallback to price-based
            support = max_put_oi_strike if max_put_oi_strike > 0 else current_price * 0.98
            resistance = max_call_oi_strike if max_call_oi_strike > 0 else current_price * 1.02
            
            return {
                'support': float(support),
                'resistance': float(resistance),
                'max_call_oi_strike': float(max_call_oi_strike),
                'max_put_oi_strike': float(max_put_oi_strike)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {
                'support': current_price * 0.98,
                'resistance': current_price * 1.02,
                'max_call_oi_strike': 0.0,
                'max_put_oi_strike': 0.0
            }
    
    def _calculate_strategy_metrics(self, oi_analysis: Dict, signal: Dict) -> Dict[str, Any]:
        """Calculate additional strategy metrics."""
        try:
            # Calculate average IV
            call_ivs = [data['iv'] for data in oi_analysis.get('calls_data', {}).values() if data['iv'] > 0]
            put_ivs = [data['iv'] for data in oi_analysis.get('puts_data', {}).values() if data['iv'] > 0]
            
            avg_call_iv = np.mean(call_ivs) if call_ivs else 0
            avg_put_iv = np.mean(put_ivs) if put_ivs else 0
            
            # Classify IV regime
            avg_iv = (avg_call_iv + avg_put_iv) / 2
            if avg_iv > 0.4:
                iv_regime = "HIGH_VOLATILITY"
            elif avg_iv > 0.25:
                iv_regime = "MEDIUM_VOLATILITY"
            else:
                iv_regime = "LOW_VOLATILITY"
            
            return {
                'avg_call_iv': avg_call_iv,
                'avg_put_iv': avg_put_iv,
                'iv_regime': iv_regime,
                'total_call_oi': sum(data['oi'] for data in oi_analysis.get('calls_data', {}).values()),
                'total_put_oi': sum(data['oi'] for data in oi_analysis.get('puts_data', {}).values()),
                'oi_skew': avg_put_iv - avg_call_iv if avg_call_iv > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy metrics: {e}")
            return {}
    
    def save_analysis_to_csv(self, analysis_result: Dict, filename: str = None) -> str:
        """Save analysis result to CSV file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"options_analysis_{analysis_result.get('index', 'UNKNOWN')}_{timestamp}.csv"
            
            filepath = os.path.join(self.results_dir, filename)
            
            # Create CSV data
            csv_data = {
                'timestamp': [analysis_result.get('timestamp', '')],
                'index': [analysis_result.get('index', '')],
                'current_price': [analysis_result.get('current_price', 0)],
                'atm_strike': [analysis_result.get('atm_strike', 0)],
                'signal_type': [analysis_result.get('signal', {}).get('signal_type', '')],
                'confidence': [analysis_result.get('signal', {}).get('confidence', 0)],
                'pcr': [analysis_result.get('oi_analysis', {}).get('pcr', 0)],
                'atm_call_oi': [analysis_result.get('oi_analysis', {}).get('atm_call_oi', 0)],
                'atm_put_oi': [analysis_result.get('oi_analysis', {}).get('atm_put_oi', 0)],
                'atm_call_oi_change': [analysis_result.get('oi_analysis', {}).get('atm_call_oi_change', 0)],
                'atm_put_oi_change': [analysis_result.get('oi_analysis', {}).get('atm_put_oi_change', 0)],
                'support': [analysis_result.get('support_resistance', {}).get('support', 0)],
                'resistance': [analysis_result.get('support_resistance', {}).get('resistance', 0)],
                'suggested_trade': [analysis_result.get('signal', {}).get('suggested_trade', '')],
                'entry_price': [analysis_result.get('signal', {}).get('entry_price', 0)],
                'stop_loss': [analysis_result.get('signal', {}).get('stop_loss', 0)],
                'target': [analysis_result.get('signal', {}).get('target', 0)],
                'iv_regime': [analysis_result.get('strategy_metrics', {}).get('iv_regime', '')]
            }
            
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"📊 Options analysis saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving analysis to CSV: {e}")
            return ""

# Global instance
enhanced_options_analyzer = EnhancedOptionsAnalyzer()
