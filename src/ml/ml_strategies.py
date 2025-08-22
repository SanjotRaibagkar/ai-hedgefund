"""
ML-Enhanced Trading Strategies Module.
Combines traditional momentum strategies with machine learning predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb

from src.strategies.eod.momentum_framework import MomentumStrategyFramework
from src.strategies.eod.momentum_indicators import MomentumIndicators
from src.ml.feature_engineering import FeatureEngineer


class MLEnhancedEODStrategy:
    """ML-enhanced EOD momentum strategy combining traditional and ML signals."""
    
    def __init__(self,
                 strategy_config: Optional[Dict[str, Any]] = None,
                 ml_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML-enhanced EOD strategy.
        
        Args:
            strategy_config: Configuration for traditional strategy
            ml_config: Configuration for ML components
        """
        self.strategy_config = strategy_config or self._get_default_strategy_config()
        self.ml_config = ml_config or self._get_default_ml_config()
        
        # Initialize components
        self.momentum_framework = MomentumStrategyFramework()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = None
        self.model_trained = False
        
        # Performance tracking
        self.predictions = []
        self.actual_returns = []
        self.model_performance = {}
        
        logger.info("MLEnhancedEODStrategy initialized with ML capabilities")
    
    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """Get default strategy configuration."""
        return {
            'momentum_weight': 0.6,
            'ml_weight': 0.4,
            'confidence_threshold': 0.7,
            'min_data_points': 100
        }
    
    def _get_default_ml_config(self) -> Dict[str, Any]:
        """Get default ML configuration."""
        return {
            'model_type': 'xgboost',
            'test_size': 0.2,
            'random_state': 42,
            'target_horizon': 5,
            'feature_selection': True
        }
    
    def _initialize_ml_model(self, model_type: str = 'xgboost'):
        """Initialize ML model based on type."""
        try:
            if model_type == 'xgboost':
                self.ml_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.ml_config['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'lightgbm':
                self.ml_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.ml_config['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'random_forest':
                self.ml_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.ml_config['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'linear':
                self.ml_model = LinearRegression()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"ML model initialized: {model_type}")
            
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
            # Fallback to linear regression
            self.ml_model = LinearRegression()
    
    def train_model(self, 
                   ticker: str,
                   start_date: str,
                   end_date: str) -> Dict[str, Any]:
        """
        Train ML model on historical data.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for training
            end_date: End date for training
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training ML model for {ticker}")
            
            # Initialize model
            self._initialize_ml_model(self.ml_config['model_type'])
            
            # Create features and target
            features, target = self.feature_engineer.create_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                target_horizon=self.ml_config['target_horizon']
            )
            
            if features.empty or target.empty:
                raise ValueError("No features or target data available")
            
            if len(features) < self.strategy_config['min_data_points']:
                raise ValueError(f"Insufficient data points: {len(features)} < {self.strategy_config['min_data_points']}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target,
                test_size=self.ml_config['test_size'],
                random_state=self.ml_config['random_state']
            )
            
            # Scale features
            X_train_scaled, y_train = self.feature_engineer.fit_transform(X_train, y_train)
            X_test_scaled = self.feature_engineer.transform(X_test)
            
            # Train model
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = self.ml_model.predict(X_train_scaled)
            y_pred_test = self.ml_model.predict(X_test_scaled)
            
            # Calculate performance metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Store performance
            self.model_performance = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # Get feature importance
            feature_importance = self.feature_engineer.get_feature_importance(self.ml_model)
            
            self.model_trained = True
            
            logger.info(f"ML model training completed for {ticker}")
            logger.info(f"Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
            
            return {
                'model_performance': self.model_performance,
                'feature_importance': feature_importance,
                'model_type': self.ml_config['model_type']
            }
            
        except Exception as e:
            logger.error(f"ML model training failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def predict_returns(self, 
                       ticker: str,
                       start_date: str,
                       end_date: str) -> Dict[str, Any]:
        """
        Predict future returns using trained ML model.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for prediction data
            end_date: End date for prediction data
            
        Returns:
            Prediction results dictionary
        """
        try:
            if not self.model_trained:
                raise ValueError("Model must be trained before making predictions")
            
            logger.info(f"Making ML predictions for {ticker}")
            
            # Create features for prediction
            features, _ = self.feature_engineer.create_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                target_horizon=self.ml_config['target_horizon']
            )
            
            if features.empty:
                raise ValueError("No features available for prediction")
            
            # Scale features
            features_scaled = self.feature_engineer.transform(features)
            
            # Make predictions
            predictions = self.ml_model.predict(features_scaled)
            
            # Create prediction DataFrame
            prediction_df = pd.DataFrame({
                'date': features.index,
                'predicted_return': predictions,
                'confidence': self._calculate_prediction_confidence(features_scaled)
            })
            
            # Store predictions
            self.predictions.append({
                'ticker': ticker,
                'predictions': prediction_df,
                'timestamp': datetime.now()
            })
            
            logger.info(f"ML predictions completed for {ticker}")
            
            return {
                'predictions': prediction_df,
                'latest_prediction': predictions[-1] if len(predictions) > 0 else None,
                'prediction_confidence': prediction_df['confidence'].iloc[-1] if len(prediction_df) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        try:
            # Simple confidence based on feature variance
            feature_variance = features.var(axis=1)
            confidence = 1 / (1 + feature_variance)
            
            # Normalize to [0, 1]
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            
            return confidence.values
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return np.ones(len(features))
    
    def analyze_stock(self, 
                     ticker: str,
                     start_date: str,
                     end_date: str) -> Dict[str, Any]:
        """
        Analyze stock using combined traditional and ML signals.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Combined analysis results
        """
        try:
            logger.info(f"Performing combined analysis for {ticker}")
            
            # Get traditional momentum analysis
            # For now, create a simple analysis structure since we don't have the full framework integration
            momentum_analysis = {
                'recommendations': [{
                    'ticker': ticker,
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'Traditional momentum analysis not fully integrated'
                }]
            }
            
            # Get ML predictions
            ml_predictions = self.predict_returns(ticker, start_date, end_date)
            
            if 'error' in ml_predictions:
                logger.warning(f"ML prediction failed, using only traditional signals: {ml_predictions['error']}")
                return momentum_analysis
            
            # Combine signals
            combined_signals = self._combine_signals(momentum_analysis, ml_predictions)
            
            # Generate final recommendation
            final_recommendation = self._generate_final_recommendation(combined_signals)
            
            logger.info(f"Combined analysis completed for {ticker}")
            
            return {
                'ticker': ticker,
                'momentum_analysis': momentum_analysis,
                'ml_predictions': ml_predictions,
                'combined_signals': combined_signals,
                'final_recommendation': final_recommendation,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Combined analysis failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _combine_signals(self, 
                        momentum_analysis: Dict[str, Any],
                        ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine traditional momentum and ML signals."""
        try:
            combined = {}
            
            # Extract momentum signal
            momentum_signal = 0
            if 'recommendations' in momentum_analysis and momentum_analysis['recommendations']:
                recommendation = momentum_analysis['recommendations'][0]
                if recommendation['action'] == 'BUY':
                    momentum_signal = recommendation.get('confidence', 0.5)
                elif recommendation['action'] == 'SELL':
                    momentum_signal = -recommendation.get('confidence', 0.5)
            
            # Extract ML signal
            ml_signal = 0
            if 'latest_prediction' in ml_predictions and ml_predictions['latest_prediction'] is not None:
                ml_signal = ml_predictions['latest_prediction']
                # Normalize ML signal to [-1, 1] range
                ml_signal = np.clip(ml_signal * 10, -1, 1)  # Scale factor of 10
            
            # Combine signals with weights
            momentum_weight = self.strategy_config['momentum_weight']
            ml_weight = self.strategy_config['ml_weight']
            
            combined_signal = (momentum_weight * momentum_signal + ml_weight * ml_signal)
            
            # Calculate combined confidence
            momentum_confidence = abs(momentum_signal)
            ml_confidence = ml_predictions.get('prediction_confidence', 0.5)
            combined_confidence = (momentum_weight * momentum_confidence + ml_weight * ml_confidence)
            
            combined = {
                'momentum_signal': momentum_signal,
                'ml_signal': ml_signal,
                'combined_signal': combined_signal,
                'momentum_confidence': momentum_confidence,
                'ml_confidence': ml_confidence,
                'combined_confidence': combined_confidence,
                'momentum_weight': momentum_weight,
                'ml_weight': ml_weight
            }
            
            return combined
            
        except Exception as e:
            logger.error(f"Signal combination failed: {e}")
            return {
                'momentum_signal': 0,
                'ml_signal': 0,
                'combined_signal': 0,
                'momentum_confidence': 0,
                'ml_confidence': 0,
                'combined_confidence': 0,
                'momentum_weight': self.strategy_config['momentum_weight'],
                'ml_weight': self.strategy_config['ml_weight']
            }
    
    def _generate_final_recommendation(self, combined_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading recommendation based on combined signals."""
        try:
            combined_signal = combined_signals['combined_signal']
            combined_confidence = combined_signals['combined_confidence']
            
            # Determine action
            if combined_signal > self.strategy_config['confidence_threshold']:
                action = 'BUY'
                strength = 'STRONG' if combined_confidence > 0.8 else 'MODERATE'
            elif combined_signal < -self.strategy_config['confidence_threshold']:
                action = 'SELL'
                strength = 'STRONG' if combined_confidence > 0.8 else 'MODERATE'
            else:
                action = 'HOLD'
                strength = 'NEUTRAL'
            
            # Calculate position size based on confidence
            if action in ['BUY', 'SELL']:
                position_size = min(combined_confidence, 1.0)
            else:
                position_size = 0.0
            
            recommendation = {
                'action': action,
                'strength': strength,
                'confidence': combined_confidence,
                'position_size': position_size,
                'combined_signal': combined_signal,
                'reasoning': self._generate_reasoning(combined_signals)
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Final recommendation generation failed: {e}")
            return {
                'action': 'HOLD',
                'strength': 'NEUTRAL',
                'confidence': 0.0,
                'position_size': 0.0,
                'combined_signal': 0.0,
                'reasoning': 'Error in recommendation generation'
            }
    
    def _generate_reasoning(self, combined_signals: Dict[str, Any]) -> str:
        """Generate reasoning for the recommendation."""
        try:
            momentum_signal = combined_signals['momentum_signal']
            ml_signal = combined_signals['ml_signal']
            combined_signal = combined_signals['combined_signal']
            
            reasoning_parts = []
            
            # Momentum reasoning
            if abs(momentum_signal) > 0.3:
                if momentum_signal > 0:
                    reasoning_parts.append("Strong positive momentum indicators")
                else:
                    reasoning_parts.append("Strong negative momentum indicators")
            
            # ML reasoning
            if abs(ml_signal) > 0.3:
                if ml_signal > 0:
                    reasoning_parts.append("ML model predicts positive returns")
                else:
                    reasoning_parts.append("ML model predicts negative returns")
            
            # Combined reasoning
            if abs(combined_signal) > 0.5:
                if combined_signal > 0:
                    reasoning_parts.append("Strong combined buy signal")
                else:
                    reasoning_parts.append("Strong combined sell signal")
            
            if not reasoning_parts:
                reasoning_parts.append("Mixed signals, neutral recommendation")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return "Unable to generate reasoning"
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategy summary."""
        return {
            'strategy_type': 'ML-Enhanced EOD Momentum',
            'model_trained': self.model_trained,
            'model_type': self.ml_config['model_type'],
            'model_performance': self.model_performance,
            'strategy_config': self.strategy_config,
            'ml_config': self.ml_config,
            'total_predictions': len(self.predictions),
            'feature_summary': self.feature_engineer.get_feature_summary()
        } 