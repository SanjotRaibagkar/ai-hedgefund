#!/usr/bin/env python3
"""
Options Analyzer for Nifty and BankNifty
Simplified but comprehensive options analysis with OI patterns, 
volatility analysis, and strike recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from src.tools.enhanced_api import get_prices, get_option_chain


class OptionsAnalyzer:
    """Options Analyzer for Nifty and BankNifty."""
    
    def __init__(self):
        """Initialize Options Analyzer."""
        self.logger = logging.getLogger(__name__)
        
    def analyze_index_options(self, index: str = 'NIFTY') -> Dict[str, Any]:
        """
        Analyze options for Nifty or BankNifty.
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            
        Returns:
            Dictionary with options analysis results
        """
        self.logger.info(f"Analyzing {index} options")
        
        try:
            # Get current index price
            index_ticker = f"{index}.NS"
            current_data = get_prices(index_ticker, 
                                    (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                                    datetime.now().strftime('%Y-%m-%d'))
            
            if current_data is None or current_data.empty:
                self.logger.error(f"Could not fetch {index} data")
                return {}
            
            current_price = current_data['close_price'].iloc[-1]
            
            # Get options chain
            options_data = get_option_chain(index_ticker)
            if not options_data:
                self.logger.error(f"Could not fetch {index} options data")
                return {}
            
            # Analyze options
            analysis = self._analyze_options_chain(options_data, current_price, index)
            
            return {
                'index': index,
                'current_price': current_price,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {index} options: {e}")
            return {}
    
    def _analyze_options_chain(self, options_data: Dict, current_price: float, index: str) -> Dict[str, Any]:
        """Analyze options chain data."""
        analysis = {
            'oi_analysis': self._analyze_oi_patterns(options_data, current_price),
            'volatility_analysis': self._analyze_volatility(options_data, current_price),
            'strike_recommendations': self._get_strike_recommendations(options_data, current_price),
            'market_sentiment': self._analyze_market_sentiment(options_data, current_price)
        }
        
        return analysis
    
    def _analyze_oi_patterns(self, options_data: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze Open Interest patterns."""
        try:
            calls_data = options_data.get('calls', [])
            puts_data = options_data.get('puts', [])
            
            # Calculate PCR (Put-Call Ratio)
            total_call_oi = sum(call.get('openInterest', 0) for call in calls_data)
            total_put_oi = sum(put.get('openInterest', 0) for put in puts_data)
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Find max OI strikes
            max_call_oi_strike = max(calls_data, key=lambda x: x.get('openInterest', 0)) if calls_data else None
            max_put_oi_strike = max(puts_data, key=lambda x: x.get('openInterest', 0)) if puts_data else None
            
            return {
                'put_call_ratio': pcr,
                'max_call_oi_strike': max_call_oi_strike.get('strikePrice', 0) if max_call_oi_strike else 0,
                'max_put_oi_strike': max_put_oi_strike.get('strikePrice', 0) if max_put_oi_strike else 0,
                'interpretation': self._interpret_pcr(pcr)
            }
            
        except Exception as e:
            self.logger.error(f"Error in OI analysis: {e}")
            return {}
    
    def _analyze_volatility(self, options_data: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze volatility patterns."""
        try:
            calls_data = options_data.get('calls', [])
            puts_data = options_data.get('puts', [])
            
            # Calculate average IV
            call_ivs = [call.get('impliedVolatility', 0) for call in calls_data if call.get('impliedVolatility', 0) > 0]
            put_ivs = [put.get('impliedVolatility', 0) for put in puts_data if put.get('impliedVolatility', 0) > 0]
            
            avg_call_iv = np.mean(call_ivs) if call_ivs else 0
            avg_put_iv = np.mean(put_ivs) if put_ivs else 0
            
            return {
                'average_call_iv': avg_call_iv,
                'average_put_iv': avg_put_iv,
                'iv_spread': avg_put_iv - avg_call_iv,
                'volatility_regime': self._classify_volatility_regime(avg_call_iv, avg_put_iv)
            }
            
        except Exception as e:
            self.logger.error(f"Error in volatility analysis: {e}")
            return {}
    
    def _get_strike_recommendations(self, options_data: Dict, current_price: float) -> Dict[str, Any]:
        """Get strike selection recommendations."""
        try:
            calls_data = options_data.get('calls', [])
            puts_data = options_data.get('puts', [])
            
            recommendations = {
                'high_iv_strikes': self._find_high_iv_strikes(calls_data, puts_data),
                'low_iv_strikes': self._find_low_iv_strikes(calls_data, puts_data),
                'atm_strikes': self._find_atm_strikes(calls_data, puts_data, current_price)
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in strike recommendations: {e}")
            return {}
    
    def _analyze_market_sentiment(self, options_data: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze market sentiment from options data."""
        try:
            calls_data = options_data.get('calls', [])
            puts_data = options_data.get('puts', [])
            
            sentiment = {
                'pcr_sentiment': self._pcr_sentiment_analysis(calls_data, puts_data),
                'iv_sentiment': self._iv_sentiment_analysis(calls_data, puts_data),
                'overall_sentiment': 'NEUTRAL'
            }
            
            # Calculate overall sentiment
            sentiment['overall_sentiment'] = self._calculate_overall_sentiment(sentiment)
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {}
    
    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret Put-Call Ratio."""
        if pcr > 1.5:
            return "EXTREME_FEAR - High put buying indicates bearish sentiment"
        elif pcr > 1.2:
            return "FEAR - Above average put buying"
        elif pcr > 0.8:
            return "NEUTRAL - Balanced options activity"
        elif pcr > 0.5:
            return "GREED - Above average call buying"
        else:
            return "EXTREME_GREED - High call buying indicates bullish sentiment"
    
    def _classify_volatility_regime(self, avg_call_iv: float, avg_put_iv: float) -> str:
        """Classify volatility regime."""
        avg_iv = (avg_call_iv + avg_put_iv) / 2
        
        if avg_iv > 0.4:
            return "HIGH_VOLATILITY"
        elif avg_iv > 0.25:
            return "MEDIUM_VOLATILITY"
        else:
            return "LOW_VOLATILITY"
    
    def _find_high_iv_strikes(self, calls_data: List, puts_data: List) -> List[Dict]:
        """Find strikes with high implied volatility."""
        try:
            all_options = calls_data + puts_data
            high_iv_options = sorted(all_options, key=lambda x: x.get('impliedVolatility', 0), reverse=True)[:5]
            
            return [{
                'strike': opt.get('strikePrice'),
                'type': 'call' if opt in calls_data else 'put',
                'iv': opt.get('impliedVolatility'),
                'oi': opt.get('openInterest')
            } for opt in high_iv_options]
            
        except Exception as e:
            self.logger.error(f"Error finding high IV strikes: {e}")
            return []
    
    def _find_low_iv_strikes(self, calls_data: List, puts_data: List) -> List[Dict]:
        """Find strikes with low implied volatility."""
        try:
            all_options = calls_data + puts_data
            low_iv_options = sorted(all_options, key=lambda x: x.get('impliedVolatility', 0))[:5]
            
            return [{
                'strike': opt.get('strikePrice'),
                'type': 'call' if opt in calls_data else 'put',
                'iv': opt.get('impliedVolatility'),
                'oi': opt.get('openInterest')
            } for opt in low_iv_options]
            
        except Exception as e:
            self.logger.error(f"Error finding low IV strikes: {e}")
            return []
    
    def _find_atm_strikes(self, calls_data: List, puts_data: List, current_price: float) -> List[Dict]:
        """Find At-The-Money strikes."""
        try:
            atm_strikes = []
            
            for call in calls_data:
                strike = call.get('strikePrice', 0)
                if 0.98 * current_price <= strike <= 1.02 * current_price:
                    atm_strikes.append({
                        'strike': strike,
                        'type': 'call',
                        'iv': call.get('impliedVolatility'),
                        'oi': call.get('openInterest'),
                        'premium': call.get('lastPrice')
                    })
            
            return sorted(atm_strikes, key=lambda x: abs(x['strike'] - current_price))[:3]
            
        except Exception as e:
            self.logger.error(f"Error finding ATM strikes: {e}")
            return []
    
    def _pcr_sentiment_analysis(self, calls_data: List, puts_data: List) -> str:
        """Analyze sentiment based on PCR."""
        try:
            total_call_oi = sum(call.get('openInterest', 0) for call in calls_data)
            total_put_oi = sum(put.get('openInterest', 0) for put in puts_data)
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            if pcr > 1.2:
                return "BEARISH"
            elif pcr < 0.8:
                return "BULLISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"Error in PCR sentiment analysis: {e}")
            return "NEUTRAL"
    
    def _iv_sentiment_analysis(self, calls_data: List, puts_data: List) -> str:
        """Analyze sentiment based on IV."""
        try:
            call_ivs = [call.get('impliedVolatility', 0) for call in calls_data if call.get('impliedVolatility', 0) > 0]
            put_ivs = [put.get('impliedVolatility', 0) for put in puts_data if put.get('impliedVolatility', 0) > 0]
            
            avg_call_iv = np.mean(call_ivs) if call_ivs else 0
            avg_put_iv = np.mean(put_ivs) if put_ivs else 0
            
            if avg_put_iv > avg_call_iv * 1.1:
                return "BEARISH"
            elif avg_call_iv > avg_put_iv * 1.1:
                return "BULLISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"Error in IV sentiment analysis: {e}")
            return "NEUTRAL"
    
    def _calculate_overall_sentiment(self, sentiment_data: Dict) -> str:
        """Calculate overall sentiment from multiple indicators."""
        try:
            sentiments = [
                sentiment_data.get('pcr_sentiment', 'NEUTRAL'),
                sentiment_data.get('iv_sentiment', 'NEUTRAL')
            ]
            
            bullish_count = sentiments.count('BULLISH')
            bearish_count = sentiments.count('BEARISH')
            
            if bullish_count > bearish_count:
                return "BULLISH"
            elif bearish_count > bullish_count:
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"Error calculating overall sentiment: {e}")
            return "NEUTRAL"
    
    def get_analyzer_summary(self) -> Dict[str, Any]:
        """Get analyzer configuration and capabilities."""
        return {
            'name': 'Options Analyzer',
            'description': 'Options analysis for Nifty and BankNifty',
            'features': [
                'OI Pattern Analysis',
                'Volatility Analysis',
                'Strike Selection Recommendations',
                'Market Sentiment Analysis'
            ],
            'supported_indices': ['NIFTY', 'BANKNIFTY'],
            'analysis_components': [
                'Put-Call Ratio Analysis',
                'IV Analysis',
                'Strike Selection',
                'Sentiment Analysis'
            ]
        } 