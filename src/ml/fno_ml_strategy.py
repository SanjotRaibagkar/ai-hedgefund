"""
FNO-Based ML Strategy Module.
Uses only FNO (Futures & Options) data from fno_bhav_copy table for enhanced ML predictions.
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
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
import duckdb

from src.tools.enhanced_api import get_fno_bhav_copy_data


class FNOMLStrategy:
    """FNO-enhanced ML strategy using only fno_bhav_copy table data."""
    
    def __init__(self,
                 strategy_config: Optional[Dict[str, Any]] = None,
                 ml_config: Optional[Dict[str, Any]] = None):
        """
        Initialize FNO-enhanced ML strategy.
        
        Args:
            strategy_config: Configuration for strategy parameters
            ml_config: Configuration for ML components
        """
        self.strategy_config = strategy_config or self._get_default_strategy_config()
        self.ml_config = ml_config or self._get_default_ml_config()
        
        # Initialize components
        self.scaler = StandardScaler()
        
        # ML models for different horizons (ensemble support)
        self.models = {
            '1d': {},
            '5d': {},
            '21d': {}
        }
        self.model_trained = {horizon: False for horizon in self.models.keys()}
        self.ensemble_models = {}
        self.model_performances = {}
        
        # Performance tracking
        self.predictions = []
        self.model_performance = {}
        
        logger.info("FNOMLStrategy initialized with FNO-only capabilities")
    
    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """Get default strategy configuration."""
        return {
            'min_data_points': 100,  # Increased for better model training
            'min_oi_threshold': 500,  # Increased minimum open interest for better liquidity
            'min_volume_threshold': 1000,  # Increased minimum volume for better liquidity
            'confidence_threshold': 0.75,  # Increased confidence threshold for better signal quality
            'price_change_threshold': 0.02,  # 2% threshold for trading signals
            'ensemble_voting': True,  # Enable ensemble voting from multiple models
            'min_liquidity_score': 0.6  # Minimum liquidity score threshold
        }
    
    def _get_default_ml_config(self) -> Dict[str, Any]:
        """Get default ML configuration."""
        return {
            'model_type': 'ensemble',  # Changed to ensemble for multiple models
            'test_size': 0.2,
            'random_state': 42,
            'prediction_horizons': [1, 5, 21],
            'feature_selection': True,
            'ensemble_models': ['xgboost', 'lightgbm', 'random_forest', 'linear'],
            'ensemble_weights': [0.4, 0.3, 0.2, 0.1]  # Weighted voting
        }
    
    def _initialize_ml_model(self, model_type: str = 'xgboost'):
        """Initialize ML model based on type."""
        try:
            if model_type == 'xgboost':
                return xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.ml_config['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'lightgbm':
                return lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.ml_config['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'random_forest':
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.ml_config['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'linear':
                return LinearRegression()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
            return LinearRegression()
    
    def _get_fno_data_for_ticker(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get FNO data for a specific ticker from the database."""
        try:
            logger.info(f"Fetching FNO data for {ticker}")
            
            # Get FNO data from database
            fno_data = get_fno_bhav_copy_data(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if fno_data.empty:
                logger.warning(f"No FNO data found for {ticker}")
                return pd.DataFrame()
            
            # Enhanced filtering with liquidity score
            fno_data = self._apply_liquidity_filters(fno_data)
            
            if fno_data.empty:
                logger.warning(f"No relevant FNO data after filtering for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(fno_data)} FNO records for {ticker}")
            return fno_data
            
        except Exception as e:
            logger.error(f"FNO data retrieval failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _apply_liquidity_filters(self, fno_data: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced liquidity filters to FNO data."""
        try:
            # Calculate liquidity score for each row
            fno_data = fno_data.copy()
            
            # Normalize OI and Volume (0-1 scale)
            max_oi = fno_data['OpnIntrst'].max()
            max_volume = fno_data['TtlTradgVol'].max()
            
            if max_oi > 0:
                fno_data['oi_normalized'] = fno_data['OpnIntrst'] / max_oi
            else:
                fno_data['oi_normalized'] = 0
                
            if max_volume > 0:
                fno_data['volume_normalized'] = fno_data['TtlTradgVol'] / max_volume
            else:
                fno_data['volume_normalized'] = 0
            
            # Calculate liquidity score (weighted average of OI and Volume)
            fno_data['liquidity_score'] = (
                0.6 * fno_data['oi_normalized'] + 
                0.4 * fno_data['volume_normalized']
            )
            
            # Apply filters
            filtered_data = fno_data[
                (fno_data['OpnIntrst'] >= self.strategy_config['min_oi_threshold']) &
                (fno_data['TtlTradgVol'] >= self.strategy_config['min_volume_threshold']) &
                (fno_data['liquidity_score'] >= self.strategy_config['min_liquidity_score'])
            ]
            
            logger.info(f"Liquidity filtering: {len(fno_data)} -> {len(filtered_data)} records")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Liquidity filtering failed: {e}")
            return fno_data
    
    def _initialize_ensemble_models(self, horizon: str):
        """Initialize ensemble models for a specific horizon."""
        try:
            ensemble_models = {}
            
            for model_type in self.ml_config['ensemble_models']:
                ensemble_models[model_type] = self._initialize_ml_model(model_type)
            
            self.models[horizon] = ensemble_models
            logger.info(f"Initialized ensemble models for {horizon}: {list(ensemble_models.keys())}")
            
        except Exception as e:
            logger.error(f"Ensemble model initialization failed for {horizon}: {e}")
    
    def _train_ensemble_models(self, horizon: str, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train ensemble models and return performance metrics."""
        try:
            ensemble_models = self.models[horizon]
            model_performances = {}
            
            for model_name, model in ensemble_models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate performance
                performance = {
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                    'test_mae': mean_absolute_error(y_test, y_pred_test)
                }
                
                model_performances[model_name] = performance
                logger.info(f"{model_name} for {horizon}: R² = {performance['test_r2']:.4f}")
            
            return model_performances
            
        except Exception as e:
            logger.error(f"Ensemble training failed for {horizon}: {e}")
            return {}
    
    def _ensemble_predict(self, horizon: str, X: pd.DataFrame) -> Tuple[float, float]:
        """Make ensemble prediction with confidence score."""
        try:
            ensemble_models = self.models[horizon]
            predictions = []
            weights = self.ml_config['ensemble_weights']
            
            for i, (model_name, model) in enumerate(ensemble_models.items()):
                pred = model.predict(X)[0]  # Single prediction
                predictions.append(pred * weights[i])
            
            # Weighted average prediction
            ensemble_prediction = sum(predictions)
            
            # Calculate confidence based on model agreement
            base_predictions = [model.predict(X)[0] for model in ensemble_models.values()]
            confidence = 1.0 - np.std(base_predictions) / (np.mean(np.abs(base_predictions)) + 1e-8)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return ensemble_prediction, confidence
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed for {horizon}: {e}")
            return 0.0, 0.0
    
    def _create_daily_fno_features(self, fno_data: pd.DataFrame) -> pd.DataFrame:
        """Create daily FNO features by combining futures and options data."""
        try:
            logger.info("Creating daily FNO features")
            
            # Group by date and calculate daily features
            daily_features = []
            
            for date in fno_data['TRADE_DATE'].unique():
                date_data = fno_data[fno_data['TRADE_DATE'] == date]
                
                # Separate futures and options data
                futures_data = date_data[date_data['OptnTp'].isna()]  # Futures have no option type
                options_data = date_data[date_data['OptnTp'].notna()]  # Options have CE/PE
                
                # Calculate base features
                features = {
                    'date': date,
                    'total_oi': date_data['OpnIntrst'].sum(),
                    'total_volume': date_data['TtlTradgVol'].sum(),
                    'total_turnover': date_data['TtlTrfVal'].sum(),
                    'num_instruments': len(date_data),
                    'avg_closing_price': date_data['ClsPric'].mean(),
                    'price_range': date_data['HghPric'].max() - date_data['LwPric'].min(),
                    'avg_strike_price': date_data['StrkPric'].mean() if 'StrkPric' in date_data.columns else 0
                }
                
                # Futures-specific features
                if not futures_data.empty:
                    features.update({
                        'futures_oi': futures_data['OpnIntrst'].sum(),
                        'futures_volume': futures_data['TtlTradgVol'].sum(),
                        'futures_turnover': futures_data['TtlTrfVal'].sum(),
                        'futures_count': len(futures_data),
                        'futures_avg_price': futures_data['ClsPric'].mean(),
                        'futures_price_range': futures_data['HghPric'].max() - futures_data['LwPric'].min()
                    })
                else:
                    features.update({
                        'futures_oi': 0, 'futures_volume': 0, 'futures_turnover': 0,
                        'futures_count': 0, 'futures_avg_price': 0, 'futures_price_range': 0
                    })
                
                # Options-specific features
                if not options_data.empty:
                    call_data = options_data[options_data['OptnTp'] == 'CE']
                    put_data = options_data[options_data['OptnTp'] == 'PE']
                    
                    features.update({
                        'options_oi': options_data['OpnIntrst'].sum(),
                        'options_volume': options_data['TtlTradgVol'].sum(),
                        'options_turnover': options_data['TtlTrfVal'].sum(),
                        'options_count': len(options_data),
                        'call_oi': call_data['OpnIntrst'].sum(),
                        'put_oi': put_data['OpnIntrst'].sum(),
                        'call_volume': call_data['TtlTradgVol'].sum(),
                        'put_volume': put_data['TtlTradgVol'].sum(),
                        'call_count': len(call_data),
                        'put_count': len(put_data),
                        'avg_call_strike': call_data['StrkPric'].mean() if not call_data.empty else 0,
                        'avg_put_strike': put_data['StrkPric'].mean() if not put_data.empty else 0
                    })
                    
                    # Calculate PCR (Put-Call Ratio)
                    features['pcr_oi'] = features['put_oi'] / features['call_oi'] if features['call_oi'] > 0 else 0
                    features['pcr_volume'] = features['put_volume'] / features['call_volume'] if features['call_volume'] > 0 else 0
                else:
                    features.update({
                        'options_oi': 0, 'options_volume': 0, 'options_turnover': 0,
                        'options_count': 0, 'call_oi': 0, 'put_oi': 0, 'call_volume': 0, 'put_volume': 0,
                        'call_count': 0, 'put_count': 0, 'avg_call_strike': 0, 'avg_put_strike': 0,
                        'pcr_oi': 0, 'pcr_volume': 0
                    })
                
                # Calculate ratios
                features['futures_oi_ratio'] = features['futures_oi'] / features['total_oi'] if features['total_oi'] > 0 else 0
                features['options_oi_ratio'] = features['options_oi'] / features['total_oi'] if features['total_oi'] > 0 else 0
                features['futures_volume_ratio'] = features['futures_volume'] / features['total_volume'] if features['total_volume'] > 0 else 0
                features['options_volume_ratio'] = features['options_volume'] / features['total_volume'] if features['total_volume'] > 0 else 0
                
                daily_features.append(features)
            
            # Create DataFrame
            features_df = pd.DataFrame(daily_features)
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df.set_index('date', inplace=True)
            
            # Clean infinite and NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0)  # Replace NaN with 0
            
            # Add rolling features
            rolling_features = [
                'total_oi', 'total_volume', 'total_turnover', 'futures_oi', 'options_oi',
                'call_oi', 'put_oi', 'pcr_oi', 'pcr_volume', 'avg_closing_price'
            ]
            
            for col in rolling_features:
                if col in features_df.columns:
                    features_df[f'{col}_sma_5'] = features_df[col].rolling(5).mean()
                    features_df[f'{col}_sma_20'] = features_df[col].rolling(20).mean()
                    features_df[f'{col}_change'] = features_df[col].pct_change()
                    features_df[f'{col}_volatility'] = features_df[col].rolling(10).std()
            
            # Add derived features
            features_df['oi_volume_ratio'] = features_df['total_oi'] / features_df['total_volume'] if 'total_volume' in features_df.columns else 0
            features_df['turnover_per_volume'] = features_df['total_turnover'] / features_df['total_volume'] if 'total_volume' in features_df.columns else 0
            
            # Add momentum features
            if 'avg_closing_price' in features_df.columns:
                features_df['price_momentum_5d'] = features_df['avg_closing_price'].pct_change(5)
                features_df['price_momentum_10d'] = features_df['avg_closing_price'].pct_change(10)
                features_df['price_momentum_20d'] = features_df['avg_closing_price'].pct_change(20)
            
            # Comprehensive data cleaning
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0)
            
            # Clip extreme values to prevent overflow
            for col in features_df.columns:
                if features_df[col].dtype in ['float64', 'float32']:
                    # Clip to reasonable range (-1000 to 1000)
                    features_df[col] = np.clip(features_df[col], -1000, 1000)
            
            logger.info(f"Created {len(features_df.columns)} daily FNO features")
            return features_df
            
        except Exception as e:
            logger.error(f"Daily FNO feature creation failed: {e}")
            return pd.DataFrame()
    
    def _create_targets(self, fno_data: pd.DataFrame, horizons: List[int]) -> Dict[str, pd.Series]:
        """Create target variables for different prediction horizons using FNO price data."""
        try:
            targets = {}
            
            # Get daily average closing prices
            daily_prices = fno_data.groupby('TRADE_DATE')['ClsPric'].mean()
            daily_prices.index = pd.to_datetime(daily_prices.index)
            daily_prices = daily_prices.sort_index()
            
            for horizon in horizons:
                # Calculate future returns
                future_returns = daily_prices.pct_change(horizon).shift(-horizon)
                targets[f'{horizon}d'] = future_returns
            
            return targets
            
        except Exception as e:
            logger.error(f"Target creation failed: {e}")
            return {}
    
    def train_models(self, 
                    ticker: str,
                    start_date: str,
                    end_date: str) -> Dict[str, Any]:
        """
        Train ML models for different prediction horizons using FNO data only.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for training
            end_date: End date for training
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training FNO-only ML models for {ticker}")
            
            # Get FNO data
            fno_data = self._get_fno_data_for_ticker(ticker, start_date, end_date)
            
            if fno_data.empty:
                raise ValueError("No FNO data available for training")
            
            # Create features
            features = self._create_daily_fno_features(fno_data)
            
            if features.empty:
                raise ValueError("No features available for training")
            
            # Create targets
            targets = self._create_targets(fno_data, self.ml_config['prediction_horizons'])
            
            # Align features and targets
            aligned_data = self._align_features_and_targets(features, targets)
            
            if aligned_data['features'].empty:
                raise ValueError("No aligned data available for training")
            
            # Train models for each horizon
            training_results = {}
            
            for horizon in self.ml_config['prediction_horizons']:
                horizon_key = f'{horizon}d'
                
                if horizon_key not in aligned_data['targets']:
                    continue
                
                logger.info(f"Training model for {horizon_key} horizon")
                
                # Get features and target for this horizon
                features = aligned_data['features']
                target = aligned_data['targets'][horizon_key]
                
                # Remove rows with NaN values
                valid_mask = ~(features.isna().any(axis=1) | target.isna())
                features = features[valid_mask]
                target = target[valid_mask]
                
                if len(features) < self.strategy_config['min_data_points']:
                    logger.warning(f"Insufficient data for {horizon_key} horizon: {len(features)} < {self.strategy_config['min_data_points']}")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target,
                    test_size=self.ml_config['test_size'],
                    random_state=self.ml_config['random_state']
                )
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Initialize and train ensemble models
                if self.ml_config['model_type'] == 'ensemble':
                    self._initialize_ensemble_models(horizon_key)
                    model_performances = self._train_ensemble_models(
                        horizon_key, X_train_scaled, y_train, X_test_scaled, y_test
                    )
                    
                    # Calculate ensemble performance (average of all models)
                    avg_performance = {
                        'train_rmse': np.mean([p['train_rmse'] for p in model_performances.values()]),
                        'test_rmse': np.mean([p['test_rmse'] for p in model_performances.values()]),
                        'train_r2': np.mean([p['train_r2'] for p in model_performances.values()]),
                        'test_r2': np.mean([p['test_r2'] for p in model_performances.values()]),
                        'train_mae': np.mean([p['train_mae'] for p in model_performances.values()]),
                        'test_mae': np.mean([p['test_mae'] for p in model_performances.values()]),
                        'training_samples': len(X_train),
                        'test_samples': len(X_test)
                    }
                    
                    performance = avg_performance
                    self.model_performances[horizon_key] = model_performances
                else:
                    # Single model training (fallback)
                    model = self._initialize_ml_model(self.ml_config['model_type'])
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Calculate performance metrics
                    performance = {
                        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                        'train_r2': r2_score(y_train, y_pred_train),
                        'test_r2': r2_score(y_test, y_pred_test),
                        'train_mae': mean_absolute_error(y_train, y_pred_train),
                        'test_mae': mean_absolute_error(y_test, y_pred_test),
                        'training_samples': len(X_train),
                        'test_samples': len(X_test)
                    }
                    
                    # Store single model
                    self.models[horizon_key] = model
                
                self.model_trained[horizon_key] = True
                self.model_performance[horizon_key] = performance
                
                # Get feature importance from best model (XGBoost for ensemble)
                if self.ml_config['model_type'] == 'ensemble':
                    best_model = self.models[horizon_key]['xgboost']
                    feature_importance = self._get_feature_importance(best_model, features.columns)
                else:
                    feature_importance = self._get_feature_importance(model, features.columns)
                
                training_results[horizon_key] = {
                    'model_performance': performance,
                    'feature_importance': feature_importance,
                    'model_type': self.ml_config['model_type']
                }
                
                logger.info(f"Model trained for {horizon_key}: R² = {performance['test_r2']:.4f}")
            
            logger.info(f"Training completed for {ticker}")
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _align_features_and_targets(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Align features and targets by date."""
        try:
            # Find common dates
            common_dates = features.index
            
            for target_name, target_series in targets.items():
                common_dates = common_dates.intersection(target_series.index)
            
            if len(common_dates) == 0:
                return {'features': pd.DataFrame(), 'targets': {}}
            
            # Align features and targets
            aligned_features = features.loc[common_dates]
            aligned_targets = {}
            
            for target_name, target_series in targets.items():
                aligned_targets[target_name] = target_series.loc[common_dates]
            
            return {
                'features': aligned_features,
                'targets': aligned_targets
            }
            
        except Exception as e:
            logger.error(f"Feature-target alignment failed: {e}")
            return {'features': pd.DataFrame(), 'targets': {}}
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}
    
    def predict_returns(self, 
                       ticker: str,
                       start_date: str,
                       end_date: str) -> Dict[str, Any]:
        """
        Predict returns for different horizons using FNO data only.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for prediction data
            end_date: End date for prediction data
            
        Returns:
            Prediction results dictionary
        """
        try:
            if not any(self.model_trained.values()):
                raise ValueError("No models trained. Please train models first.")
            
            logger.info(f"Making FNO-only predictions for {ticker}")
            
            # Get FNO data
            fno_data = self._get_fno_data_for_ticker(ticker, start_date, end_date)
            
            if fno_data.empty:
                raise ValueError("No FNO data available for prediction")
            
            # Create features
            features = self._create_daily_fno_features(fno_data)
            
            if features.empty:
                raise ValueError("No features available for prediction")
            
            # Make predictions for each horizon
            predictions = {}
            
            for horizon_key, model in self.models.items():
                if not self.model_trained[horizon_key] or model is None:
                    continue
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make ensemble predictions with confidence
                if self.ml_config['model_type'] == 'ensemble' and isinstance(model, dict):
                    horizon_predictions = []
                    confidence_scores = []
                    
                    for i, (date_idx, feature_row) in enumerate(features.iterrows()):
                        feature_row_scaled = self.scaler.transform(feature_row.values.reshape(1, -1))
                        pred, confidence = self._ensemble_predict(horizon_key, feature_row_scaled)
                        horizon_predictions.append(pred)
                        confidence_scores.append(confidence)
                    
                    horizon_predictions = np.array(horizon_predictions)
                    confidence_scores = np.array(confidence_scores)
                else:
                    # Single model prediction (fallback)
                    horizon_predictions = model.predict(features_scaled)
                    confidence_scores = self._calculate_prediction_confidence(features_scaled)
                
                # Create prediction DataFrame with enhanced confidence filtering
                pred_df = pd.DataFrame({
                    'date': features.index,
                    'predicted_return': horizon_predictions,
                    'confidence': confidence_scores
                })
                
                # Apply confidence threshold filtering
                high_confidence_mask = pred_df['confidence'] >= self.strategy_config['confidence_threshold']
                filtered_predictions = pred_df[high_confidence_mask].copy()
                
                predictions[horizon_key] = {
                    'predictions': filtered_predictions,
                    'all_predictions': pred_df,  # Keep all predictions for analysis
                    'latest_prediction': horizon_predictions[-1] if len(horizon_predictions) > 0 else None,
                    'prediction_confidence': confidence_scores[-1] if len(confidence_scores) > 0 else None,
                    'high_confidence_count': high_confidence_mask.sum(),
                    'total_predictions': len(pred_df)
                }
            
            # Store predictions
            self.predictions.append({
                'ticker': ticker,
                'predictions': predictions,
                'timestamp': datetime.now()
            })
            
            logger.info(f"Predictions completed for {ticker}")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        try:
            # Simple confidence based on feature variance
            feature_variance = np.var(features, axis=1)
            confidence = 1 / (1 + feature_variance)
            
            # Normalize to [0, 1]
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return np.ones(features.shape[0])
    
    def analyze_stock(self, 
                     ticker: str,
                     start_date: str,
                     end_date: str) -> Dict[str, Any]:
        """
        Analyze stock using FNO-only ML predictions.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Analysis results
        """
        try:
            logger.info(f"Performing FNO-only analysis for {ticker}")
            
            # Get predictions
            predictions = self.predict_returns(ticker, start_date, end_date)
            
            if 'error' in predictions:
                return {'error': predictions['error']}
            
            # Generate trading signals for each horizon
            signals = {}
            
            for horizon_key, pred_data in predictions.items():
                if 'latest_prediction' not in pred_data:
                    continue
                
                latest_pred = pred_data['latest_prediction']
                confidence = pred_data.get('prediction_confidence', 0.5)
                
                # Generate signal based on prediction with enhanced confidence filtering
                threshold = self.strategy_config['price_change_threshold']
                confidence_threshold = self.strategy_config['confidence_threshold']
                
                # Only generate signals if confidence meets threshold
                if confidence >= confidence_threshold:
                    if latest_pred > threshold:
                        action = 'BUY'
                        strength = 'STRONG' if confidence > 0.85 else 'MODERATE'
                    elif latest_pred < -threshold:
                        action = 'SELL'
                        strength = 'STRONG' if confidence > 0.85 else 'MODERATE'
                    else:
                        action = 'HOLD'
                        strength = 'NEUTRAL'
                else:
                    # Low confidence - default to HOLD
                    action = 'HOLD'
                    strength = 'LOW_CONFIDENCE'
                
                signals[horizon_key] = {
                    'action': action,
                    'strength': strength,
                    'confidence': confidence,
                    'predicted_return': latest_pred,
                    'reasoning': f"FNO-only ML model predicts {latest_pred*100:.2f}% return over {horizon_key}"
                }
            
            # Generate overall recommendation
            overall_signal = self._generate_overall_signal(signals)
            
            logger.info(f"FNO-only analysis completed for {ticker}")
            
            return {
                'ticker': ticker,
                'predictions': predictions,
                'signals': signals,
                'overall_signal': overall_signal,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"FNO-only analysis failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _generate_overall_signal(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall trading signal from multiple horizons."""
        try:
            if not signals:
                return {
                    'action': 'HOLD',
                    'strength': 'NEUTRAL',
                    'confidence': 0.0,
                    'reasoning': 'No signals available'
                }
            
            # Count signals by action
            action_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_confidence = 0
            total_predictions = 0
            
            for horizon_key, signal in signals.items():
                action_counts[signal['action']] += 1
                total_confidence += signal['confidence']
                total_predictions += 1
            
            # Determine overall action
            if action_counts['BUY'] > action_counts['SELL'] and action_counts['BUY'] > action_counts['HOLD']:
                action = 'BUY'
            elif action_counts['SELL'] > action_counts['BUY'] and action_counts['SELL'] > action_counts['HOLD']:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # Calculate overall confidence
            overall_confidence = total_confidence / total_predictions if total_predictions > 0 else 0
            
            # Determine strength
            if overall_confidence > 0.8:
                strength = 'STRONG'
            elif overall_confidence > 0.5:
                strength = 'MODERATE'
            else:
                strength = 'NEUTRAL'
            
            reasoning = f"Overall signal based on {total_predictions} horizon predictions: {action_counts}"
            
            return {
                'action': action,
                'strength': strength,
                'confidence': overall_confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Overall signal generation failed: {e}")
            return {
                'action': 'HOLD',
                'strength': 'NEUTRAL',
                'confidence': 0.0,
                'reasoning': 'Error in signal generation'
            }
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategy summary."""
        return {
            'strategy_type': 'FNO-Only ML Strategy',
            'models_trained': self.model_trained,
            'model_performance': self.model_performance,
            'strategy_config': self.strategy_config,
            'ml_config': self.ml_config,
            'total_predictions': len(self.predictions),
            'prediction_horizons': self.ml_config['prediction_horizons']
        }
