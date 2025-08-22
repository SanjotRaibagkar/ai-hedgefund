"""
Model Manager Module for ML Strategy Management.
Manages multiple ML models, model selection, and ensemble methods.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb

from src.ml.feature_engineering import FeatureEngineer
from src.ml.mlflow_tracker import MLflowTracker


class MLModelManager:
    """Manages multiple ML models and ensemble methods."""
    
    def __init__(self,
                 models_config: Optional[Dict[str, Any]] = None,
                 ensemble_config: Optional[Dict[str, Any]] = None,
                 mlflow_tracker: Optional[MLflowTracker] = None):
        """
        Initialize ML model manager.
        
        Args:
            models_config: Configuration for individual models
            ensemble_config: Configuration for ensemble methods
            mlflow_tracker: MLflow tracker instance
        """
        self.models_config = models_config or self._get_default_models_config()
        self.ensemble_config = ensemble_config or self._get_default_ensemble_config()
        self.mlflow_tracker = mlflow_tracker
        
        # Model storage
        self.models = {}
        self.ensemble_model = None
        self.model_performance = {}
        self.feature_engineer = FeatureEngineer()
        
        # Model registry
        self.model_registry_path = "./models/ml_model_manager"
        self._setup_model_registry()
        
        logger.info("MLModelManager initialized with model management capabilities")
    
    def _get_default_models_config(self) -> Dict[str, Any]:
        """Get default models configuration."""
        return {
            'xgboost': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'lightgbm': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'random_forest': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'linear': {
                'enabled': True,
                'params': {}
            }
        }
    
    def _get_default_ensemble_config(self) -> Dict[str, Any]:
        """Get default ensemble configuration."""
        return {
            'enabled': True,
            'method': 'voting',
            'weights': None,  # Equal weights by default
            'voting': 'soft'
        }
    
    def _setup_model_registry(self):
        """Setup model registry directory."""
        try:
            os.makedirs(self.model_registry_path, exist_ok=True)
            logger.info(f"Model registry setup: {self.model_registry_path}")
            
        except Exception as e:
            logger.error(f"Model registry setup failed: {e}")
    
    def train_all_models(self,
                        ticker: str,
                        start_date: str,
                        end_date: str,
                        test_size: float = 0.2,
                        random_state: int = 42) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for training
            end_date: End date for training
            test_size: Test set size
            random_state: Random state for reproducibility
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training all models for {ticker}")
            
            # Start MLflow experiment if tracker is available
            run_id = None
            if self.mlflow_tracker:
                run_id = self.mlflow_tracker.start_experiment(
                    run_name=f"model_training_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags={'ticker': ticker, 'training_type': 'all_models'}
                )
            
            # Create features and target
            features, target = self.feature_engineer.create_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if features.empty or target.empty:
                raise ValueError("No features or target data available")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target,
                test_size=test_size,
                random_state=random_state
            )
            
            # Scale features
            X_train_scaled, y_train = self.feature_engineer.fit_transform(X_train, y_train)
            X_test_scaled = self.feature_engineer.transform(X_test)
            
            # Train individual models
            training_results = {}
            trained_models = {}
            
            for model_name, model_config in self.models_config.items():
                if not model_config.get('enabled', True):
                    continue
                
                try:
                    logger.info(f"Training {model_name} for {ticker}")
                    
                    # Train model
                    model, performance = self._train_single_model(
                        model_name, model_config, X_train_scaled, y_train, X_test_scaled, y_test
                    )
                    
                    if model is not None:
                        trained_models[model_name] = model
                        training_results[model_name] = performance
                        
                        # Log to MLflow if available
                        if self.mlflow_tracker:
                            self._log_training_to_mlflow(
                                model, model_name, model_name, 
                                {'samples': len(X_train), 'features': X_train.shape[1]},
                                performance.get('feature_importance', {})
                            )
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    training_results[model_name] = {'error': str(e)}
            
            # Store trained models
            self.models = trained_models
            
            # Create ensemble if enabled
            ensemble_result = None
            if self.ensemble_config.get('enabled', True) and len(trained_models) > 1:
                try:
                    logger.info(f"Creating ensemble model for {ticker}")
                    ensemble_result = self._create_ensemble(trained_models, X_train_scaled, y_train, X_test_scaled, y_test)
                    
                    if self.mlflow_tracker:
                        self._log_training_to_mlflow(
                            self.ensemble_model, 'ensemble', 'ensemble',
                            {'samples': len(X_train), 'features': X_train.shape[1]},
                            {}
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to create ensemble: {e}")
                    ensemble_result = {'error': str(e)}
            
            # Store performance
            self.model_performance = training_results
            if ensemble_result:
                self.model_performance['ensemble'] = ensemble_result
            
            # End MLflow experiment
            if self.mlflow_tracker and run_id:
                self.mlflow_tracker.end_experiment()
            
            # Save models
            self._save_models(ticker)
            
            logger.info(f"All models training completed for {ticker}")
            return {
                'individual_models': training_results,
                'ensemble': ensemble_result,
                'total_models_trained': len(trained_models)
            }
            
        except Exception as e:
            logger.error(f"All models training failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _train_single_model(self,
                           model_name: str,
                           model_config: Dict[str, Any],
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_test: pd.DataFrame,
                           y_test: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """Train a single model."""
        try:
            # Initialize model
            model = self._initialize_model(model_name, model_config['params'])
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate performance metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Get feature importance
            feature_importance = self.feature_engineer.get_feature_importance(model)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            performance = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'training_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"{model_name} training completed - Test R²: {test_r2:.4f}")
            return model, performance
            
        except Exception as e:
            logger.error(f"Single model training failed for {model_name}: {e}")
            return None, {'error': str(e)}
    
    def _initialize_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Initialize a model based on name and parameters."""
        try:
            if model_name == 'xgboost':
                return xgb.XGBRegressor(**params)
            elif model_name == 'lightgbm':
                return lgb.LGBMRegressor(**params)
            elif model_name == 'random_forest':
                return RandomForestRegressor(**params)
            elif model_name == 'linear':
                return LinearRegression(**params)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
        except Exception as e:
            logger.error(f"Model initialization failed for {model_name}: {e}")
            raise
    
    def _create_ensemble(self,
                        models: Dict[str, Any],
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series) -> Dict[str, Any]:
        """Create ensemble model."""
        try:
            # Create voting regressor
            estimators = [(name, model) for name, model in models.items()]
            
            ensemble_config = self.ensemble_config
            weights = ensemble_config.get('weights')
            voting = ensemble_config.get('voting', 'soft')
            
            self.ensemble_model = VotingRegressor(
                estimators=estimators,
                weights=weights,
                n_jobs=-1
            )
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = self.ensemble_model.predict(X_train)
            y_pred_test = self.ensemble_model.predict(X_test)
            
            # Calculate performance metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.ensemble_model, X_train, y_train, cv=5, scoring='r2')
            
            performance = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'ensemble_method': self.ensemble_config.get('method', 'voting'),
                'ensemble_weights': weights,
                'training_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Ensemble model training completed - Test R²: {test_r2:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return {'error': str(e)}
    
    def predict_with_all_models(self,
                               ticker: str,
                               start_date: str,
                               end_date: str) -> Dict[str, Any]:
        """
        Make predictions with all trained models.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for prediction data
            end_date: End date for prediction data
            
        Returns:
            Predictions from all models
        """
        try:
            if not self.models:
                raise ValueError("No models trained. Run train_all_models first.")
            
            logger.info(f"Making predictions with all models for {ticker}")
            
            # Create features for prediction
            features, _ = self.feature_engineer.create_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if features.empty:
                raise ValueError("No features available for prediction")
            
            # Scale features
            features_scaled = self.feature_engineer.transform(features)
            
            # Make predictions with all models
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    model_predictions = model.predict(features_scaled)
                    predictions[model_name] = {
                        'predictions': model_predictions,
                        'latest_prediction': model_predictions[-1] if len(model_predictions) > 0 else None
                    }
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = {'error': str(e)}
            
            # Make ensemble prediction if available
            if self.ensemble_model:
                try:
                    ensemble_predictions = self.ensemble_model.predict(features_scaled)
                    predictions['ensemble'] = {
                        'predictions': ensemble_predictions,
                        'latest_prediction': ensemble_predictions[-1] if len(ensemble_predictions) > 0 else None
                    }
                    
                except Exception as e:
                    logger.error(f"Ensemble prediction failed: {e}")
                    predictions['ensemble'] = {'error': str(e)}
            
            logger.info(f"All models predictions completed for {ticker}")
            return predictions
            
        except Exception as e:
            logger.error(f"All models prediction failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def get_best_model(self, metric: str = 'test_r2') -> Tuple[str, Any]:
        """
        Get the best performing model based on metric.
        
        Args:
            metric: Performance metric to use for comparison
            
        Returns:
            Tuple of (model_name, model)
        """
        try:
            if not self.model_performance:
                raise ValueError("No model performance data available")
            
            # Filter out models with errors
            valid_models = {
                name: perf for name, perf in self.model_performance.items()
                if 'error' not in perf and metric in perf
            }
            
            if not valid_models:
                raise ValueError(f"No valid models with metric {metric}")
            
            # Find best model
            best_model_name = max(valid_models.keys(), 
                                key=lambda x: valid_models[x][metric])
            best_model = self.models.get(best_model_name)
            
            logger.info(f"Best model by {metric}: {best_model_name}")
            return best_model_name, best_model
            
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None, None
    
    def _save_models(self, ticker: str):
        """Save trained models to disk."""
        try:
            # Create ticker-specific directory
            ticker_dir = os.path.join(self.model_registry_path, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Save individual models
            for model_name, model in self.models.items():
                model_path = os.path.join(ticker_dir, f"{model_name}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save ensemble model
            if self.ensemble_model:
                ensemble_path = os.path.join(ticker_dir, "ensemble_model.pkl")
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(self.ensemble_model, f)
            
            # Save performance metrics
            performance_path = os.path.join(ticker_dir, "performance_metrics.json")
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2, default=str)
            
            # Save feature engineer
            feature_engineer_path = os.path.join(ticker_dir, "feature_engineer.pkl")
            with open(feature_engineer_path, 'wb') as f:
                pickle.dump(self.feature_engineer, f)
            
            logger.info(f"Models saved for {ticker} in {ticker_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models for {ticker}: {e}")
    
    def load_models(self, ticker: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if models loaded successfully
        """
        try:
            ticker_dir = os.path.join(self.model_registry_path, ticker)
            
            if not os.path.exists(ticker_dir):
                logger.warning(f"No saved models found for {ticker}")
                return False
            
            # Load individual models
            self.models = {}
            for model_name in self.models_config.keys():
                model_path = os.path.join(ticker_dir, f"{model_name}_model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            # Load ensemble model
            ensemble_path = os.path.join(ticker_dir, "ensemble_model.pkl")
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
            
            # Load performance metrics
            performance_path = os.path.join(ticker_dir, "performance_metrics.json")
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
            
            # Load feature engineer
            feature_engineer_path = os.path.join(ticker_dir, "feature_engineer.pkl")
            if os.path.exists(feature_engineer_path):
                with open(feature_engineer_path, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
            
            logger.info(f"Models loaded for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models for {ticker}: {e}")
            return False
    
    def _log_training_to_mlflow(self,
                               model: Any,
                               model_name: str,
                               model_type: str,
                               training_data_info: Dict[str, Any],
                               feature_importance: Dict[str, float]):
        """Log model training to MLflow."""
        try:
            if self.mlflow_tracker:
                self.mlflow_tracker.log_model_training(
                    model=model,
                    model_name=model_name,
                    model_type=model_type,
                    training_data_info=training_data_info,
                    feature_importance=feature_importance
                )
                
        except Exception as e:
            logger.error(f"Failed to log training to MLflow: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model manager summary."""
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'ensemble_available': self.ensemble_model is not None,
            'ensemble_config': self.ensemble_config,
            'model_performance': self.model_performance,
            'feature_engineer_summary': self.feature_engineer.get_feature_summary(),
            'model_registry_path': self.model_registry_path,
            'mlflow_tracker_available': self.mlflow_tracker is not None
        } 