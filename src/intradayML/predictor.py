"""
Intraday ML Predictor
Makes predictions using trained ML models for intraday trading.
"""

import sys
import os
sys.path.append('./src')

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .feature_engineer import IntradayFeatureEngineer
from .model_trainer import IntradayMLTrainer


class IntradayPredictor:
    """Makes predictions using trained ML models for intraday trading."""
    
    def __init__(self, models_dir: str = "models/intraday_ml"):
        """
        Initialize the predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.feature_engineer = IntradayFeatureEngineer()
        self.model_trainer = IntradayMLTrainer(models_dir)
        
        # Loaded models and metadata
        self.loaded_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performances = {}
        
        logger.info("🚀 Intraday ML Predictor initialized")
    
    def load_models_for_index(self, index_symbol: str) -> bool:
        """
        Load trained models for a specific index.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            logger.info(f"📥 Loading models for {index_symbol}")
            
            # Load models using model trainer
            success = self.model_trainer.load_models(index_symbol)
            
            if success:
                # Copy models and metadata
                self.loaded_models[index_symbol] = self.model_trainer.models.copy()
                self.scalers[index_symbol] = self.model_trainer.scalers.get(index_symbol)
                self.feature_importance[index_symbol] = self.model_trainer.feature_importance.copy()
                self.model_performances[index_symbol] = self.model_trainer.model_performances.copy()
                
                logger.info(f"✅ Models loaded for {index_symbol}")
                return True
            else:
                logger.warning(f"⚠️ Failed to load models for {index_symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading models for {index_symbol}: {e}")
            return False
    
    def predict_single(self, index_symbol: str, timestamp: datetime, model_name: str = None) -> Dict[str, Any]:
        """
        Make a single prediction for a specific timestamp.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            timestamp: Timestamp for prediction
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info(f"🔮 Making prediction for {index_symbol} at {timestamp}")
            
            # Check if models are loaded
            if index_symbol not in self.loaded_models:
                logger.warning(f"⚠️ Models not loaded for {index_symbol}, attempting to load...")
                if not self.load_models_for_index(index_symbol):
                    return {}
            
            # Create features
            features = self.feature_engineer.create_complete_features(index_symbol, timestamp)
            
            if not features:
                logger.warning(f"⚠️ No features available for {index_symbol} at {timestamp}")
                return {}
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, index_symbol)
            
            if feature_vector is None:
                logger.warning(f"⚠️ Failed to prepare feature vector for {index_symbol}")
                return {}
            
            # Select model
            if model_name is None:
                model_name = self._get_best_model(index_symbol)
            
            if model_name not in self.loaded_models[index_symbol]:
                logger.error(f"❌ Model {model_name} not found for {index_symbol}")
                return {}
            
            model = self.loaded_models[index_symbol][model_name]
            
            # Make prediction
            prediction_result = self._make_prediction(model, feature_vector, index_symbol, model_name)
            
            # Add metadata
            prediction_result.update({
                'index_symbol': index_symbol,
                'timestamp': timestamp,
                'model_used': model_name,
                'features_used': len(feature_vector)
            })
            
            logger.info(f"✅ Prediction completed for {index_symbol}: {prediction_result['prediction']}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return {}
    
    def _prepare_feature_vector(self, features: Dict[str, float], index_symbol: str) -> Optional[np.ndarray]:
        """Prepare feature vector for prediction."""
        try:
            # Get feature columns
            feature_columns = self.model_trainer.get_feature_columns()
            
            # Create feature vector
            feature_vector = []
            for col in feature_columns:
                if col in features:
                    feature_vector.append(features[col])
                else:
                    feature_vector.append(0.0)  # Default value for missing features
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaling if available
            if index_symbol in self.scalers and self.scalers[index_symbol] is not None:
                scaler = self.scalers[index_symbol]
                feature_vector = scaler.transform(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Error preparing feature vector: {e}")
            return None
    
    def _get_best_model(self, index_symbol: str) -> str:
        """Get the best performing model for an index."""
        try:
            if index_symbol not in self.model_performances:
                return 'random_forest'  # Default fallback
            
            performances = self.model_performances[index_symbol]
            
            # Find model with highest F1 score
            best_model = None
            best_f1 = -1
            
            for model_name, performance in performances.items():
                if performance['f1_score'] > best_f1:
                    best_f1 = performance['f1_score']
                    best_model = model_name
            
            return best_model if best_model else 'random_forest'
            
        except Exception as e:
            logger.error(f"❌ Error getting best model: {e}")
            return 'random_forest'
    
    def _make_prediction(self, model: Any, feature_vector: np.ndarray, index_symbol: str, model_name: str) -> Dict[str, Any]:
        """Make prediction using a specific model."""
        try:
            # Get prediction and probability
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            
            # Map prediction to label
            if prediction == 1:
                direction = "UP"
                confidence = probabilities[1]  # Probability of UP
            else:
                direction = "DOWN"
                confidence = probabilities[0]  # Probability of DOWN
            
            # Get feature importance if available
            feature_importance = {}
            if index_symbol in self.feature_importance and model_name in self.feature_importance[index_symbol]:
                feature_importance = self.feature_importance[index_symbol][model_name]
            
            result = {
                'prediction': prediction,
                'direction': direction,
                'confidence': confidence,
                'probabilities': {
                    'UP': probabilities[1],
                    'DOWN': probabilities[0]
                },
                'feature_importance': feature_importance
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error making prediction with {model_name}: {e}")
            return {}
    
    def predict_batch(self, index_symbol: str, start_timestamp: datetime, end_timestamp: datetime, model_name: str = None) -> pd.DataFrame:
        """
        Make batch predictions for a time range.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            start_timestamp: Start timestamp for predictions
            end_timestamp: End timestamp for predictions
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            DataFrame with batch predictions
        """
        try:
            logger.info(f"📊 Making batch predictions for {index_symbol} from {start_timestamp} to {end_timestamp}")
            
            # Get all timestamps in the range
            timestamps = self._get_timestamps_in_range(index_symbol, start_timestamp, end_timestamp)
            
            if timestamps.empty:
                logger.warning(f"⚠️ No timestamps found for {index_symbol} in the specified range")
                return pd.DataFrame()
            
            # Make predictions for each timestamp
            predictions = []
            
            for _, row in timestamps.iterrows():
                timestamp = row['timestamp']
                
                prediction_result = self.predict_single(index_symbol, timestamp, model_name)
                
                if prediction_result:
                    predictions.append(prediction_result)
            
            if predictions:
                predictions_df = pd.DataFrame(predictions)
                logger.info(f"✅ Batch predictions completed: {len(predictions_df)} predictions")
                return predictions_df
            else:
                logger.warning("⚠️ No predictions generated")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error making batch predictions: {e}")
            return pd.DataFrame()
    
    def _get_timestamps_in_range(self, index_symbol: str, start_timestamp: datetime, end_timestamp: datetime) -> pd.DataFrame:
        """Get all timestamps in the specified range."""
        try:
            query = """
                SELECT DISTINCT timestamp FROM intraday_index_data 
                WHERE index_symbol = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            timestamps = self.feature_engineer.conn.execute(query, [index_symbol, start_timestamp, end_timestamp]).fetchdf()
            
            return timestamps
            
        except Exception as e:
            logger.error(f"❌ Error getting timestamps: {e}")
            return pd.DataFrame()
    
    def predict_current(self, index_symbol: str, model_name: str = None) -> Dict[str, Any]:
        """
        Make prediction for the current market conditions.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            Dictionary with current prediction
        """
        try:
            current_time = datetime.now()
            
            logger.info(f"🔮 Making current prediction for {index_symbol}")
            
            # Check if market is open
            if not self._is_market_open():
                logger.warning("⚠️ Market is closed, prediction may not be accurate")
            
            # Make prediction
            prediction_result = self.predict_single(index_symbol, current_time, model_name)
            
            if prediction_result:
                prediction_result['prediction_time'] = current_time
                prediction_result['market_status'] = 'OPEN' if self._is_market_open() else 'CLOSED'
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Error making current prediction: {e}")
            return {}
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            
            # Market hours: 9:30 AM to 3:30 PM IST
            if "09:30" <= current_time <= "15:30":
                # Check if it's a weekday
                if now.weekday() < 5:  # Monday = 0, Friday = 4
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking market status: {e}")
            return False
    
    def get_prediction_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for batch predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Dictionary with prediction summary
        """
        try:
            if predictions_df.empty:
                return {}
            
            summary = {
                'total_predictions': len(predictions_df),
                'up_predictions': len(predictions_df[predictions_df['direction'] == 'UP']),
                'down_predictions': len(predictions_df[predictions_df['direction'] == 'DOWN']),
                'avg_confidence': predictions_df['confidence'].mean(),
                'confidence_std': predictions_df['confidence'].std(),
                'prediction_distribution': predictions_df['direction'].value_counts().to_dict()
            }
            
            # Time-based analysis
            if 'timestamp' in predictions_df.columns:
                predictions_df['hour'] = predictions_df['timestamp'].dt.hour
                summary['predictions_by_hour'] = predictions_df['hour'].value_counts().to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating prediction summary: {e}")
            return {}
    
    def get_feature_importance_summary(self, index_symbol: str, model_name: str = None) -> Dict[str, float]:
        """
        Get feature importance summary for a model.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            model_name: Specific model (if None, uses best model)
            
        Returns:
            Dictionary with feature importance
        """
        try:
            if model_name is None:
                model_name = self._get_best_model(index_symbol)
            
            if (index_symbol in self.feature_importance and 
                model_name in self.feature_importance[index_symbol]):
                
                importance = self.feature_importance[index_symbol][model_name]
                
                # Sort by importance
                sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                
                return sorted_importance
            else:
                logger.warning(f"⚠️ No feature importance available for {index_symbol} - {model_name}")
                return {}
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {e}")
            return {}
    
    def validate_prediction(self, index_symbol: str, timestamp: datetime, actual_direction: str) -> Dict[str, Any]:
        """
        Validate a prediction against actual market movement.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            timestamp: Timestamp of the prediction
            actual_direction: Actual market direction ("UP" or "DOWN")
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Get the prediction that was made
            prediction_result = self.predict_single(index_symbol, timestamp)
            
            if not prediction_result:
                return {}
            
            # Compare prediction with actual
            predicted_direction = prediction_result['direction']
            is_correct = predicted_direction == actual_direction
            
            validation_result = {
                'timestamp': timestamp,
                'index_symbol': index_symbol,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'is_correct': is_correct,
                'confidence': prediction_result['confidence'],
                'model_used': prediction_result['model_used']
            }
            
            logger.info(f"✅ Validation: Predicted {predicted_direction}, Actual {actual_direction}, Correct: {is_correct}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Error validating prediction: {e}")
            return {}
    
    def get_model_performance(self, index_symbol: str) -> Dict[str, Any]:
        """
        Get performance metrics for all models for an index.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            
        Returns:
            Dictionary with model performances
        """
        try:
            if index_symbol not in self.model_performances:
                logger.warning(f"⚠️ No performance data available for {index_symbol}")
                return {}
            
            return self.model_performances[index_symbol]
            
        except Exception as e:
            logger.error(f"❌ Error getting model performance: {e}")
            return {}
    
    def close(self):
        """Close resources."""
        try:
            self.feature_engineer.close()
            self.model_trainer.close()
            logger.info("🔒 Predictor resources closed")
        except Exception as e:
            logger.error(f"❌ Error closing resources: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
