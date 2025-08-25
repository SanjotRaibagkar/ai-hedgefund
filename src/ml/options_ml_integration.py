#!/usr/bin/env python3
"""
Options ML Integration
Integrates options analysis signals with ML models for enhanced predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from loguru import logger

from src.nsedata.NseUtility import NseUtils


class OptionsMLIntegration:
    """Integrates options analysis with ML models."""
    
    def __init__(self):
        """Initialize Options ML Integration."""
        self.logger = logging.getLogger(__name__)
        self.nse = NseUtils()
        
    def get_options_signals(self, indices: List[str] = ['NIFTY', 'BANKNIFTY']) -> Dict[str, Any]:
        """
        Get options signals for ML integration.
        
        Args:
            indices: List of indices to analyze
            
        Returns:
            Dictionary with options signals
        """
        signals = {}
        
        for index in indices:
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
                    signal_strength = 0.0
                    
                    # Strategy Rules
                    if pcr > 0.9 and atm_put_oi_change > 0 and atm_call_oi_change < 0:
                        signal = "BULLISH"
                        confidence = min(90, 60 + (pcr - 0.9) * 100)
                        signal_strength = (confidence - 50) / 40  # Normalize to 0-1
                    elif pcr < 0.8 and atm_call_oi_change > 0 and atm_put_oi_change < 0:
                        signal = "BEARISH"
                        confidence = min(90, 60 + (0.8 - pcr) * 100)
                        signal_strength = -(confidence - 50) / 40  # Negative for bearish
                    elif 0.8 <= pcr <= 1.2 and atm_call_oi_change > 0 and atm_put_oi_change > 0:
                        signal = "RANGE"
                        confidence = 70.0
                        signal_strength = 0.0  # Neutral for range
                    
                    signals[index] = {
                        'current_price': current_price,
                        'atm_strike': atm_strike,
                        'pcr': pcr,
                        'signal': signal,
                        'confidence': confidence,
                        'signal_strength': signal_strength,
                        'atm_call_oi': atm_call_oi,
                        'atm_put_oi': atm_put_oi,
                        'atm_call_oi_change': atm_call_oi_change,
                        'atm_put_oi_change': atm_put_oi_change,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                else:
                    self.logger.warning(f"No options data available for {index}")
                    
            except Exception as e:
                self.logger.error(f"Error getting options signals for {index}: {e}")
        
        return signals
    
    def enhance_ml_features(self, base_features: pd.DataFrame, options_signals: Dict[str, Any]) -> pd.DataFrame:
        """
        Enhance ML features with options signals.
        
        Args:
            base_features: Base features DataFrame
            options_signals: Options signals dictionary
            
        Returns:
            Enhanced features DataFrame
        """
        enhanced_features = base_features.copy()
        
        # Add options signals as features
        for index, signal in options_signals.items():
            enhanced_features[f'{index}_pcr'] = signal['pcr']
            enhanced_features[f'{index}_signal_strength'] = signal['signal_strength']
            enhanced_features[f'{index}_confidence'] = signal['confidence']
            enhanced_features[f'{index}_atm_call_oi'] = signal['atm_call_oi']
            enhanced_features[f'{index}_atm_put_oi'] = signal['atm_put_oi']
            enhanced_features[f'{index}_atm_call_oi_change'] = signal['atm_call_oi_change']
            enhanced_features[f'{index}_atm_put_oi_change'] = signal['atm_put_oi_change']
            
            # Add signal type as categorical feature
            signal_mapping = {'BULLISH': 1, 'BEARISH': -1, 'RANGE': 0, 'NEUTRAL': 0}
            enhanced_features[f'{index}_signal_encoded'] = signal_mapping.get(signal['signal'], 0)
        
        return enhanced_features
    
    def get_market_sentiment_score(self, options_signals: Dict[str, Any]) -> float:
        """
        Calculate overall market sentiment score from options signals.
        
        Args:
            options_signals: Options signals dictionary
            
        Returns:
            Market sentiment score (-1 to 1, where -1 is bearish, 1 is bullish)
        """
        if not options_signals:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for index, signal in options_signals.items():
            # Weight by confidence
            weight = signal['confidence'] / 100.0
            score = signal['signal_strength'] * weight
            
            total_score += score
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def adjust_ml_prediction(self, base_prediction: float, options_signals: Dict[str, Any], 
                           adjustment_factor: float = 0.1) -> float:
        """
        Adjust ML prediction based on options signals.
        
        Args:
            base_prediction: Base ML prediction
            options_signals: Options signals dictionary
            adjustment_factor: How much to adjust the prediction (0-1)
            
        Returns:
            Adjusted prediction
        """
        sentiment_score = self.get_market_sentiment_score(options_signals)
        
        # Adjust prediction based on sentiment
        adjustment = sentiment_score * adjustment_factor
        adjusted_prediction = base_prediction * (1 + adjustment)
        
        return adjusted_prediction
    
    def get_ml_recommendations(self, base_features: pd.DataFrame, options_signals: Dict[str, Any],
                             base_prediction: float) -> Dict[str, Any]:
        """
        Get comprehensive ML recommendations with options integration.
        
        Args:
            base_features: Base features DataFrame
            options_signals: Options signals dictionary
            base_prediction: Base ML prediction
            
        Returns:
            Comprehensive recommendations
        """
        # Enhance features
        enhanced_features = self.enhance_ml_features(base_features, options_signals)
        
        # Get sentiment score
        sentiment_score = self.get_market_sentiment_score(options_signals)
        
        # Adjust prediction
        adjusted_prediction = self.adjust_ml_prediction(base_prediction, options_signals)
        
        # Generate recommendations
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'base_prediction': base_prediction,
            'adjusted_prediction': adjusted_prediction,
            'sentiment_score': sentiment_score,
            'options_signals': options_signals,
            'enhanced_features_count': len(enhanced_features.columns),
            'recommendation': self._generate_recommendation(adjusted_prediction, sentiment_score, options_signals)
        }
        
        return recommendations
    
    def _generate_recommendation(self, prediction: float, sentiment: float, 
                               options_signals: Dict[str, Any]) -> str:
        """Generate trading recommendation based on prediction and sentiment."""
        if sentiment > 0.3:
            if prediction > 0:
                return "STRONG_BUY - Options sentiment bullish, ML prediction positive"
            else:
                return "BUY - Options sentiment bullish, consider ML prediction"
        elif sentiment < -0.3:
            if prediction < 0:
                return "STRONG_SELL - Options sentiment bearish, ML prediction negative"
            else:
                return "SELL - Options sentiment bearish, consider ML prediction"
        else:
            if prediction > 0.1:
                return "BUY - Neutral options sentiment, ML prediction positive"
            elif prediction < -0.1:
                return "SELL - Neutral options sentiment, ML prediction negative"
            else:
                return "HOLD - Neutral signals from both options and ML"
