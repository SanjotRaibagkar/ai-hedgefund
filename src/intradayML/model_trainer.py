"""
Intraday ML Model Trainer
Trains and manages ML models for intraday prediction.
"""

import sys
import os
sys.path.append('./src')

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb

from .feature_engineer import IntradayFeatureEngineer


class IntradayMLTrainer:
    """Trains and manages ML models for intraday prediction."""
    
    def __init__(self, models_dir: str = "models/intraday_ml"):
        """
        Initialize the model trainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        self.feature_engineer = IntradayFeatureEngineer()
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performances = {}
        
        logger.info("🚀 Intraday ML Model Trainer initialized")
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for training."""
        return [
            # Options Chain Features
            'atm_ce_delta', 'atm_ce_theta', 'atm_ce_vega', 'atm_ce_gamma', 'atm_ce_iv', 'atm_ce_premium', 'atm_ce_oi', 'atm_ce_volume',
            'atm_pe_delta', 'atm_pe_theta', 'atm_pe_vega', 'atm_pe_gamma', 'atm_pe_iv', 'atm_pe_premium', 'atm_pe_oi', 'atm_pe_volume',
            'atm_ce_oi_change', 'atm_pe_oi_change', 'atm_oi_change_ratio',
            'otm_ce_oi_change', 'otm_pe_oi_change', 'otm_oi_change_ratio',
            'pcr_oi', 'pcr_volume', 'pcr_ratio',
            'atm_ce_iv', 'atm_pe_iv', 'atm_iv_skew', 'atm_iv_ratio',
            
            # Index Technical Features
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'turnover',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_deviation',
            'vwap', 'vwap_deviation',
            'return_15min', 'momentum_5', 'momentum_10',
            
            # Market Sentiment Features
            'fii_buy', 'fii_sell', 'fii_net', 'dii_buy', 'dii_sell', 'dii_net', 'fii_dii_ratio',
            'vix_value', 'vix_change'
        ]
    
    def prepare_training_data(self, index_symbol: str, start_date: date, end_date: date) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for model training.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            logger.info(f"📊 Preparing training data for {index_symbol}")
            
            # Get training data from feature engineer
            training_data = self.feature_engineer.get_training_data(index_symbol, start_date, end_date)
            
            if training_data.empty:
                logger.warning(f"⚠️ No training data available for {index_symbol}")
                return pd.DataFrame(), pd.Series()
            
            # Get feature columns
            feature_columns = self.get_feature_columns()
            
            # Filter available features
            available_features = [col for col in feature_columns if col in training_data.columns]
            
            # Prepare features and labels
            X = training_data[available_features].copy()
            y = training_data['label'].copy()
            
            # Handle missing values
            X = X.fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            # Remove rows with missing labels
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            logger.info(f"✅ Prepared training data: {len(X)} samples, {len(available_features)} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def initialize_models(self):
        """Initialize ML models for training."""
        try:
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            # LightGBM
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            
            # Logistic Regression
            self.models['logistic_regression'] = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
            
            # SVM (for smaller datasets)
            self.models['svm'] = SVC(
                C=1.0,
                kernel='rbf',
                random_state=42,
                probability=True
            )
            
            logger.info(f"✅ Initialized {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
    
    def train_models(self, index_symbol: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Train all models for the specified index and date range.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"🎯 Training models for {index_symbol}")
            
            # Initialize models
            self.initialize_models()
            
            # Prepare training data
            X, y = self.prepare_training_data(index_symbol, start_date, end_date)
            
            if X.empty or y.empty:
                logger.error("❌ No training data available")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[index_symbol] = scaler
            
            # Train each model
            training_results = {}
            
            for model_name, model in self.models.items():
                try:
                    logger.info(f"🏋️ Training {model_name}...")
                    
                    # Train model
                    if model_name == 'svm' and len(X_train) > 10000:
                        # Skip SVM for large datasets
                        logger.info(f"⏭️ Skipping {model_name} for large dataset")
                        continue
                    
                    if model_name in ['logistic_regression', 'svm']:
                        # Use scaled data for linear models
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        # Use original data for tree-based models
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Store performance
                    self.model_performances[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                    
                    # Get feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[model_name] = dict(zip(X.columns, model.feature_importances_))
                    
                    training_results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                    
                    logger.info(f"✅ {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error training {model_name}: {e}")
                    continue
            
            # Save models
            self.save_models(index_symbol)
            
            # Generate training report
            report = self.generate_training_report(training_results, index_symbol)
            
            logger.info(f"✅ Training completed for {index_symbol}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error in model training: {e}")
            return {}
    
    def save_models(self, index_symbol: str):
        """Save trained models to disk."""
        try:
            model_dir = os.path.join(self.models_dir, index_symbol)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            for model_name, model in self.models.items():
                model_path = os.path.join(model_dir, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scaler
            if index_symbol in self.scalers:
                scaler_path = os.path.join(model_dir, "scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[index_symbol], f)
            
            # Save feature importance
            if self.feature_importance:
                importance_path = os.path.join(model_dir, "feature_importance.pkl")
                with open(importance_path, 'wb') as f:
                    pickle.dump(self.feature_importance, f)
            
            # Save model performances
            if self.model_performances:
                performance_path = os.path.join(model_dir, "model_performances.pkl")
                with open(performance_path, 'wb') as f:
                    pickle.dump(self.model_performances, f)
            
            logger.info(f"✅ Models saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    def load_models(self, index_symbol: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            model_dir = os.path.join(self.models_dir, index_symbol)
            
            if not os.path.exists(model_dir):
                logger.warning(f"⚠️ Model directory not found: {model_dir}")
                return False
            
            # Load models
            for model_name in ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting', 'logistic_regression', 'svm']:
                model_path = os.path.join(model_dir, f"{model_name}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[index_symbol] = pickle.load(f)
            
            # Load feature importance
            importance_path = os.path.join(model_dir, "feature_importance.pkl")
            if os.path.exists(importance_path):
                with open(importance_path, 'rb') as f:
                    self.feature_importance = pickle.load(f)
            
            # Load model performances
            performance_path = os.path.join(model_dir, "model_performances.pkl")
            if os.path.exists(performance_path):
                with open(performance_path, 'rb') as f:
                    self.model_performances = pickle.load(f)
            
            logger.info(f"✅ Models loaded from {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def generate_training_report(self, training_results: Dict[str, Any], index_symbol: str) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        try:
            report = {
                'index_symbol': index_symbol,
                'training_date': datetime.now().isoformat(),
                'models_trained': len(training_results),
                'model_performances': {},
                'best_model': None,
                'feature_importance': {}
            }
            
            # Model performances
            best_f1 = 0
            for model_name, results in training_results.items():
                report['model_performances'][model_name] = {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score']
                }
                
                if results['f1_score'] > best_f1:
                    best_f1 = results['f1_score']
                    report['best_model'] = model_name
            
            # Feature importance (from best model)
            if report['best_model'] and report['best_model'] in self.feature_importance:
                report['feature_importance'] = self.feature_importance[report['best_model']]
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating training report: {e}")
            return {}
    
    def hyperparameter_tuning(self, index_symbol: str, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            model_name: Name of the model to tune
            X: Feature matrix
            y: Target variable
            
        Returns:
            Best model after tuning
        """
        try:
            logger.info(f"🔧 Performing hyperparameter tuning for {model_name}")
            
            # Define parameter grids
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'lightgbm': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
            
            if model_name not in param_grids:
                logger.warning(f"⚠️ No parameter grid defined for {model_name}")
                return None
            
            # Initialize base model
            base_models = {
                'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
                'xgboost': xgb.XGBClassifier(random_state=42, n_jobs=-1),
                'lightgbm': lgb.LGBMClassifier(random_state=42, n_jobs=-1)
            }
            
            base_model = base_models[model_name]
            param_grid = param_grids[model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            logger.info(f"✅ Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"✅ Best score: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"❌ Error in hyperparameter tuning: {e}")
            return None
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        try:
            logger.info(f"🔄 Performing {cv_folds}-fold cross-validation for {model_name}")
            
            if model_name not in self.models:
                logger.error(f"❌ Model {model_name} not found")
                return {}
            
            model = self.models[model_name]
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_weighted')
            
            results = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'min_score': cv_scores.min(),
                'max_score': cv_scores.max(),
                'all_scores': cv_scores.tolist()
            }
            
            logger.info(f"✅ CV Results for {model_name}: Mean={results['mean_score']:.3f} ± {results['std_score']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in cross-validation: {e}")
            return {}
    
    def close(self):
        """Close resources."""
        try:
            self.feature_engineer.close()
            logger.info("🔒 Model trainer resources closed")
        except Exception as e:
            logger.error(f"❌ Error closing resources: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
