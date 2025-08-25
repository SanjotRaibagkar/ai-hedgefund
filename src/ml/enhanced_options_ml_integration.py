#!/usr/bin/env python3
"""
Enhanced Options ML Integration
Integrates options analysis signals with real ML models for enhanced predictions.
Improved sentiment logic to handle conflicting signals better.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from loguru import logger

from src.nsedata.NseUtility import NseUtils
from src.ml.model_manager import MLModelManager
from src.ml.feature_engineering import FeatureEngineer
from src.tools.enhanced_api import get_prices


class EnhancedOptionsMLIntegration:
    """Enhanced integration of options analysis with real ML models."""
    
    def __init__(self):
        """Initialize Enhanced Options ML Integration."""
        self.logger = logging.getLogger(__name__)
        self.nse = NseUtils()
        self.model_manager = MLModelManager()
        self.feature_engineer = FeatureEngineer()
        
        # Initialize models for NIFTY and BANKNIFTY
        self.models_initialized = False
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models for prediction."""
        try:
            # Check if models exist, if not create sample models
            if not self.models_initialized:
                logger.info("Initializing ML models for market prediction")
                
                # For now, we'll create a simple model using recent price data
                # In production, this would load pre-trained models
                self._create_sample_models()
                self.models_initialized = True
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def _create_sample_models(self):
        """Create sample ML models using recent market data."""
        try:
            # Get recent price data for NIFTY and BANKNIFTY
            indices = ['NIFTY', 'BANKNIFTY']
            
            for index in indices:
                try:
                    # Get recent price data
                    ticker = f"{index}.NS"
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=60)  # 2 months of data
                    
                    price_data = get_prices(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    
                    if price_data is not None and not price_data.empty:
                        # Create features
                        features = self._create_features_from_prices(price_data)
                        
                        # Create target (next day return)
                        price_data['next_day_return'] = price_data['close_price'].shift(-1) / price_data['close_price'] - 1
                        target = price_data['next_day_return'].dropna()
                        
                        # Align features with target
                        features = features.iloc[:-1]  # Remove last row since we don't have next day return
                        target = target.iloc[:-1]  # Remove last row
                        
                        if len(features) > 10 and len(target) > 10:
                            # Train a simple model
                            self._train_simple_model(index, features, target)
                            
                except Exception as e:
                    logger.warning(f"Could not create model for {index}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to create sample models: {e}")
    
    def _create_features_from_prices(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from price data."""
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['close_price'] = price_data['close_price']
            features['volume'] = price_data['volume']
            
            # Technical indicators
            features['sma_5'] = price_data['close_price'].rolling(5).mean()
            features['sma_20'] = price_data['close_price'].rolling(20).mean()
            features['rsi'] = self._calculate_rsi(price_data['close_price'])
            features['volatility'] = price_data['close_price'].pct_change().rolling(10).std()
            
            # Price momentum
            features['price_momentum_1d'] = price_data['close_price'].pct_change(1)
            features['price_momentum_5d'] = price_data['close_price'].pct_change(5)
            features['volume_momentum'] = price_data['volume'].pct_change(5)
            
            # Remove NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to create features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices))
    
    def _train_simple_model(self, index: str, features: pd.DataFrame, target: pd.Series):
        """Train a simple ML model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Store model
            if not hasattr(self, 'ml_models'):
                self.ml_models = {}
            self.ml_models[index] = model
            
            logger.info(f"Simple ML model trained for {index}")
            
        except Exception as e:
            logger.error(f"Failed to train simple model for {index}: {e}")
    
    def get_options_signals(self, indices: List[str] = ['NIFTY', 'BANKNIFTY']) -> Dict[str, Any]:
        """
        Get enhanced options signals with real ML integration.
        
        Args:
            indices: List of indices to analyze
            
        Returns:
            Dictionary with enhanced options signals
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
                    
                    # Enhanced signal generation with better logic
                    signal, confidence, signal_strength = self._generate_enhanced_signal(
                        pcr, atm_call_oi_change, atm_put_oi_change, atm_call_oi, atm_put_oi
                    )
                    
                    # Get ML prediction
                    ml_prediction = self._get_ml_prediction(index, current_price)
                    
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
                        'ml_prediction': ml_prediction,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                else:
                    self.logger.warning(f"No options data available for {index}")
                    
            except Exception as e:
                self.logger.error(f"Error getting options signals for {index}: {e}")
        
        return signals
    
    def _generate_enhanced_signal(self, pcr: float, call_oi_change: float, put_oi_change: float, 
                                 call_oi: float, put_oi: float) -> tuple:
        """Generate enhanced signal with better logic."""
        try:
            # Enhanced PCR analysis
            if pcr > 1.5:
                base_signal = "BULLISH"
                base_confidence = 80
            elif pcr > 1.2:
                base_signal = "BULLISH"
                base_confidence = 70
            elif pcr < 0.7:
                base_signal = "BEARISH"
                base_confidence = 80
            elif pcr < 0.9:
                base_signal = "BEARISH"
                base_confidence = 70
            else:
                base_signal = "NEUTRAL"
                base_confidence = 50
            
            # OI change analysis
            oi_signal = "NEUTRAL"
            if call_oi_change > 0 and put_oi_change < 0:
                oi_signal = "BULLISH"
            elif call_oi_change < 0 and put_oi_change > 0:
                oi_signal = "BEARISH"
            
            # Combine signals
            if base_signal == oi_signal:
                final_signal = base_signal
                confidence = min(95, base_confidence + 10)
            elif base_signal == "NEUTRAL":
                final_signal = oi_signal
                confidence = 60
            elif oi_signal == "NEUTRAL":
                final_signal = base_signal
                confidence = base_confidence
            else:
                # Conflicting signals - use PCR as primary
                final_signal = base_signal
                confidence = base_confidence - 10
            
            # Calculate signal strength
            if final_signal == "BULLISH":
                signal_strength = (confidence - 50) / 45  # Normalize to 0-1
            elif final_signal == "BEARISH":
                signal_strength = -(confidence - 50) / 45  # Normalize to -1-0
            else:
                signal_strength = 0.0
            
            return final_signal, confidence, signal_strength
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signal: {e}")
            return "NEUTRAL", 50, 0.0
    
    def _get_ml_prediction(self, index: str, current_price: float) -> Dict[str, Any]:
        """Get ML prediction for the index."""
        try:
            if hasattr(self, 'ml_models') and index in self.ml_models:
                model = self.ml_models[index]
                
                # Create current features
                features = self._create_current_features(index, current_price)
                
                if not features.empty:
                    # Make prediction
                    prediction = model.predict([features.iloc[-1]])[0]
                    
                    return {
                        'prediction': prediction,
                        'direction': 'BULLISH' if prediction > 0 else 'BEARISH' if prediction < 0 else 'NEUTRAL',
                        'confidence': min(90, abs(prediction) * 100),
                        'model_type': 'RandomForest'
                    }
            
            # Fallback to simple prediction based on recent trend
            return self._get_simple_prediction(index, current_price)
            
        except Exception as e:
            self.logger.error(f"Error getting ML prediction for {index}: {e}")
            return {'prediction': 0, 'direction': 'NEUTRAL', 'confidence': 50, 'model_type': 'fallback'}
    
    def _create_current_features(self, index: str, current_price: float) -> pd.DataFrame:
        """Create current features for ML prediction."""
        try:
            # Get recent price data
            ticker = f"{index}.NS"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            price_data = get_prices(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if price_data is not None and not price_data.empty:
                return self._create_features_from_prices(price_data)
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error creating current features: {e}")
            return pd.DataFrame()
    
    def _get_simple_prediction(self, index: str, current_price: float) -> Dict[str, Any]:
        """Get simple prediction based on recent trend."""
        try:
            # Get recent price data
            ticker = f"{index}.NS"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)
            
            price_data = get_prices(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if price_data is not None and not price_data.empty:
                # Calculate simple momentum
                recent_prices = price_data['close_price'].tail(5)
                momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                
                return {
                    'prediction': momentum,
                    'direction': 'BULLISH' if momentum > 0.01 else 'BEARISH' if momentum < -0.01 else 'NEUTRAL',
                    'confidence': min(70, abs(momentum) * 1000),
                    'model_type': 'momentum'
                }
            
            return {'prediction': 0, 'direction': 'NEUTRAL', 'confidence': 50, 'model_type': 'fallback'}
            
        except Exception as e:
            self.logger.error(f"Error getting simple prediction: {e}")
            return {'prediction': 0, 'direction': 'NEUTRAL', 'confidence': 50, 'model_type': 'fallback'}
    
    def get_enhanced_market_sentiment_score(self, options_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enhanced market sentiment score with better conflict resolution.
        
        Args:
            options_signals: Options signals dictionary
            
        Returns:
            Enhanced sentiment analysis
        """
        if not options_signals:
            return {'overall_score': 0.0, 'sentiment': 'NEUTRAL', 'confidence': 50, 'conflicts': []}
        
        try:
            # Analyze individual signals
            signal_analysis = {}
            conflicts = []
            
            for index, signal in options_signals.items():
                # Options sentiment
                options_sentiment = signal['signal_strength']
                options_confidence = signal['confidence']
                
                # ML sentiment
                ml_prediction = signal.get('ml_prediction', {})
                ml_sentiment = ml_prediction.get('prediction', 0)
                ml_confidence = ml_prediction.get('confidence', 50)
                
                # Check for conflicts
                if (options_sentiment > 0.3 and ml_sentiment < -0.01) or (options_sentiment < -0.3 and ml_sentiment > 0.01):
                    conflicts.append(f"{index}: Options {signal['signal']} vs ML {ml_prediction.get('direction', 'NEUTRAL')}")
                
                signal_analysis[index] = {
                    'options_sentiment': options_sentiment,
                    'options_confidence': options_confidence,
                    'ml_sentiment': ml_sentiment,
                    'ml_confidence': ml_confidence,
                    'combined_score': (options_sentiment * options_confidence + ml_sentiment * ml_confidence) / (options_confidence + ml_confidence)
                }
            
            # Calculate weighted overall score
            total_weight = 0
            total_score = 0
            
            for index, analysis in signal_analysis.items():
                weight = (analysis['options_confidence'] + analysis['ml_confidence']) / 2
                total_score += analysis['combined_score'] * weight
                total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0
            
            # Determine sentiment with conflict awareness
            if len(conflicts) > 0:
                if abs(overall_score) > 0.3:
                    sentiment = "BULLISH" if overall_score > 0 else "BEARISH"
                    confidence = min(80, abs(overall_score) * 100)
                else:
                    sentiment = "NEUTRAL"
                    confidence = 50
            else:
                if overall_score > 0.2:
                    sentiment = "BULLISH"
                    confidence = min(90, overall_score * 100)
                elif overall_score < -0.2:
                    sentiment = "BEARISH"
                    confidence = min(90, abs(overall_score) * 100)
                else:
                    sentiment = "NEUTRAL"
                    confidence = 50
            
            return {
                'overall_score': overall_score,
                'sentiment': sentiment,
                'confidence': confidence,
                'conflicts': conflicts,
                'signal_analysis': signal_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced sentiment: {e}")
            return {'overall_score': 0.0, 'sentiment': 'NEUTRAL', 'confidence': 50, 'conflicts': []}
    
    def get_enhanced_recommendations(self, options_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get enhanced recommendations combining options and ML signals.
        
        Args:
            options_signals: Options signals dictionary
            
        Returns:
            Enhanced recommendations
        """
        try:
            sentiment_analysis = self.get_enhanced_market_sentiment_score(options_signals)
            
            recommendations = {
                'timestamp': datetime.now().isoformat(),
                'overall_sentiment': sentiment_analysis['sentiment'],
                'sentiment_score': sentiment_analysis['overall_score'],
                'confidence': sentiment_analysis['confidence'],
                'conflicts_detected': len(sentiment_analysis['conflicts']) > 0,
                'conflicts': sentiment_analysis['conflicts'],
                'recommendations': []
            }
            
            # Generate recommendations for each index
            for index, signal in options_signals.items():
                options_signal = signal['signal']
                ml_prediction = signal.get('ml_prediction', {})
                ml_direction = ml_prediction.get('direction', 'NEUTRAL')
                
                # Determine recommendation
                if options_signal == ml_direction:
                    recommendation = f"STRONG_{options_signal.upper()}"
                    reason = f"Both options ({options_signal}) and ML ({ml_direction}) agree"
                elif options_signal == "NEUTRAL":
                    recommendation = ml_direction
                    reason = f"Options neutral, ML suggests {ml_direction}"
                elif ml_direction == "NEUTRAL":
                    recommendation = options_signal
                    reason = f"ML neutral, options suggest {options_signal}"
                else:
                    # Conflicting signals
                    recommendation = "HOLD"
                    reason = f"Conflicting signals: Options {options_signal}, ML {ml_direction}"
                
                recommendations['recommendations'].append({
                    'index': index,
                    'recommendation': recommendation,
                    'reason': reason,
                    'options_signal': options_signal,
                    'ml_direction': ml_direction,
                    'confidence': signal['confidence']
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced recommendations: {e}")
            return {'error': str(e)}
