#!/usr/bin/env python3
"""
FNO Probability Models
Machine learning models for FNO probability prediction across multiple horizons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from loguru import logger
from datetime import datetime, timedelta
import os
import pickle
import json
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from ..models.data_models import HorizonType, FNOData, ProbabilityResult
from ..core.data_processor import FNODataProcessor


class FNOProbabilityModels:
    """Machine learning models for FNO probability prediction."""
    
    def __init__(self, models_dir: str = "models/fno_ml"):
        """Initialize the ML models."""
        self.logger = logger
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_processor = FNODataProcessor()
        
        # Model storage
        self.models = {
            HorizonType.DAILY: None,
            HorizonType.WEEKLY: None,
            HorizonType.MONTHLY: None
        }
        
        # Scalers for each horizon
        self.scalers = {
            HorizonType.DAILY: StandardScaler(),
            HorizonType.WEEKLY: StandardScaler(),
            HorizonType.MONTHLY: StandardScaler()
        }
        
        # Feature names will be dynamically set from data_processor
        self.feature_names = []
        
        # Model parameters for each horizon
        self.model_params = {
            HorizonType.DAILY: {
                'model_type': 'random_forest',
                'params': {
                    'n_estimators': 50,  # Reduced from 100
                    'max_depth': 5,      # Reduced from 6
                    'random_state': 42
                }
            },
            HorizonType.WEEKLY: {
                'model_type': 'random_forest',
                'params': {
                    'n_estimators': 50,  # Reduced from 200
                    'max_depth': 5,      # Reduced from 10
                    'random_state': 42
                }
            },
            HorizonType.MONTHLY: {
                'model_type': 'random_forest',
                'params': {
                    'n_estimators': 50,  # Reduced from 150
                    'max_depth': 5,      # Reduced from 8
                    'random_state': 42
                }
            }
        }
        
        # Model metadata
        self.model_metadata = {}
    
    def train_models(self, symbols: Optional[List[str]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train ML models for all horizons."""
        try:
            self.logger.info("Starting ML model training...")
            
            # Get training data
            if df is None:
                df = self.data_processor.get_fno_data(symbols, start_date, end_date)
            
            if df is None or len(df) == 0:
                raise ValueError("No training data available")
            
            self.logger.info(f"Training data shape: {df.shape}")
            
            results = {}
            
            # Train models for each horizon
            for horizon in HorizonType:
                self.logger.info(f"Training {horizon.value} model...")
                
                horizon_result = self._train_horizon_model(df, horizon)
                results[horizon.value] = horizon_result
                
                self.logger.info(f"✅ {horizon.value} model trained successfully")
            
            # Save models
            self._save_models()
            
            self.logger.info("🎉 All ML models trained successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to train models: {e}")
            raise
    
    def _train_horizon_model(self, df: pd.DataFrame, horizon: HorizonType) -> Dict[str, Any]:
        """Train model for a specific horizon."""
        try:
            # Prepare features and labels
            X, y = self._prepare_training_data(df, horizon)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError(f"No valid training data for {horizon.value}")
            
            # Set feature names from the prepared data
            if len(self.feature_names) == 0:
                # Get feature names from data processor
                X_df, _ = self.data_processor.prepare_features(df, horizon)
                self.feature_names = list(X_df.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = self.scalers[horizon]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get model configuration
            model_config = self.model_params[horizon]
            model_type = model_config['model_type']
            params = model_config['params']
            
            # Create and train model
            model = self._create_model(model_type, params)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Store model
            self.models[horizon] = model
            
            # Create result
            result = {
                'horizon': horizon.value,
                'model_type': model_type,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_importance': self._get_feature_importance(model, model_type),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to train {horizon.value} model: {e}")
            raise
    
    def _prepare_training_data(self, df: pd.DataFrame, horizon: HorizonType) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        try:
            # Use the same feature preparation as data_processor
            X, y = self.data_processor.prepare_features(df, horizon)
            
            if len(X) == 0 or len(y) == 0:
                return np.array([]), np.array([])
            
            # Convert to numpy arrays
            X_array = X.values
            y_array = y.values
            
            return X_array, y_array
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            return np.array([]), np.array([])
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create ML model based on type."""
        if model_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_feature_importance(self, model, model_type: str) -> Dict[str, float]:
        """Get feature importance from model."""
        try:
            if model_type == 'xgboost':
                importance = model.feature_importances_
            elif model_type == 'random_forest':
                importance = model.feature_importances_
            elif model_type == 'gradient_boosting':
                importance = model.feature_importances_
            else:
                return {}
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(importance):
                    feature_importance[feature] = float(importance[i])
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def predict_probability(self, features: Dict[str, float], horizon: HorizonType) -> Dict[str, float]:
        """Predict probability for given features."""
        try:
            model = self.models[horizon]
            scaler = self.scalers[horizon]
            
            if model is None:
                raise ValueError(f"Model for {horizon.value} not trained")
            
            # Prepare feature vector
            feature_vector = []
            for feature in self.feature_names:
                feature_vector.append(features.get(feature, 0.0))
            
            # Scale features
            X_scaled = scaler.transform([feature_vector])
            
            # Get prediction probabilities
            proba = model.predict_proba(X_scaled)[0]
            
            # Return probabilities
            return {
                'down': float(proba[0]),
                'up': float(proba[1])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to predict probability: {e}")
            return {'down': 0.5, 'up': 0.5}
    
    def load_models(self):
        """Load trained models from disk."""
        try:
            self.logger.info("Loading ML models...")
            
            for horizon in HorizonType:
                model_file = self.models_dir / f"{horizon.value}_model.pkl"
                scaler_file = self.models_dir / f"{horizon.value}_scaler.pkl"
                metadata_file = self.models_dir / f"{horizon.value}_metadata.json"
                
                if model_file.exists() and scaler_file.exists():
                    # Load model
                    with open(model_file, 'rb') as f:
                        self.models[horizon] = pickle.load(f)
                    
                    # Load scaler
                    with open(scaler_file, 'rb') as f:
                        self.scalers[horizon] = pickle.load(f)
                    
                    # Load metadata
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            self.model_metadata[horizon.value] = json.load(f)
                    
                    self.logger.info(f"✅ Loaded {horizon.value} model")
                else:
                    self.logger.warning(f"⚠️ No trained model found for {horizon.value}")
            
            self.logger.info("ML models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            self.logger.info("Saving ML models...")
            
            for horizon in HorizonType:
                model = self.models[horizon]
                scaler = self.scalers[horizon]
                
                if model is not None:
                    # Save model
                    model_file = self.models_dir / f"{horizon.value}_model.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    
                    # Save scaler
                    scaler_file = self.models_dir / f"{horizon.value}_scaler.pkl"
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    # Save metadata
                    metadata_file = self.models_dir / f"{horizon.value}_metadata.json"
                    metadata = {
                        'horizon': horizon.value,
                        'model_type': self.model_params[horizon]['model_type'],
                        'feature_names': self.feature_names,
                        'trained_at': datetime.now().isoformat(),
                        'feature_count': len(self.feature_names)
                    }
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    self.logger.info(f"✅ Saved {horizon.value} model")
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        try:
            status = {}
            
            for horizon in HorizonType:
                model = self.models[horizon]
                metadata = self.model_metadata.get(horizon.value, {})
                
                status[horizon.value] = {
                    'trained': model is not None,
                    'model_type': metadata.get('model_type', 'unknown'),
                    'trained_at': metadata.get('trained_at', 'unknown'),
                    'feature_count': metadata.get('feature_count', 0)
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get model status: {e}")
            return {}
    
    def retrain_model(self, horizon: HorizonType, **kwargs) -> Dict[str, Any]:
        """Retrain a specific model."""
        try:
            self.logger.info(f"Retraining {horizon.value} model...")
            
            # Get fresh data
            df = self.data_processor.get_fno_data()
            
            if df is None or len(df) == 0:
                raise ValueError("No data available for retraining")
            
            # Retrain model
            result = self._train_horizon_model(df, horizon)
            
            # Save updated model
            self._save_models()
            
            self.logger.info(f"✅ {horizon.value} model retrained successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrain {horizon.value} model: {e}")
            raise
    
    def evaluate_model(self, horizon: HorizonType, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate a specific model."""
        try:
            model = self.models[horizon]
            if model is None:
                raise ValueError(f"Model for {horizon.value} not trained")
            
            # Get test data
            if test_data is None:
                test_data = self.data_processor.get_fno_data()
            
            if test_data is None or len(test_data) == 0:
                raise ValueError("No test data available")
            
            # Prepare test data
            X_test, y_test = self._prepare_training_data(test_data, horizon)
            
            if len(X_test) == 0:
                raise ValueError("No valid test data")
            
            # Scale features
            scaler = self.scalers[horizon]
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            evaluation = {
                'horizon': horizon.value,
                'accuracy': accuracy,
                'test_samples': len(X_test),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {horizon.value} model: {e}")
            raise
