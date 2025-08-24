"""
Feature Engineering Module for ML-Enhanced Trading Strategies.
Creates advanced features from technical and fundamental data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.strategies.eod.momentum_indicators import MomentumIndicators
from src.tools.enhanced_api import get_financial_metrics, get_prices


class FeatureEngineer:
    """Advanced feature engineering for ML-enhanced trading strategies."""
    
    def __init__(self,
                 feature_config: Optional[Dict[str, Any]] = None,
                 scaler_type: str = 'robust'):
        """
        Initialize feature engineer.
        
        Args:
            feature_config: Configuration for feature engineering
            scaler_type: Type of scaler ('standard', 'robust')
        """
        self.feature_config = feature_config or self._get_default_config()
        self.scaler_type = scaler_type
        
        # Initialize components
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        
        # Feature storage
        self.feature_names = []
        self.feature_importance = {}
        self.scaler_fitted = False
        
        # Initialize momentum indicators
        self.momentum_indicators = MomentumIndicators()
        
        logger.info("FeatureEngineer initialized with advanced feature engineering capabilities")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            'technical_features': {
                'price_features': True,
                'volume_features': True,
                'momentum_features': True,
                'volatility_features': True,
                'trend_features': True
            },
            'fundamental_features': {
                'valuation_features': False,  # Disabled since fundamental data not available
                'financial_features': False   # Disabled since fundamental data not available
            },
            'derived_features': {
                'interaction_features': True,
                'lag_features': True,
                'rolling_features': True
            }
        }
    
    def create_features(self, 
                       ticker: str,
                       start_date: str,
                       end_date: str,
                       target_column: str = 'close_price',
                       target_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create comprehensive feature set for ML modeling.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            target_column: Target variable column
            target_horizon: Prediction horizon in days
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            logger.info(f"Creating features for {ticker} from {start_date} to {end_date}")
            
            # Get base data
            prices_df = get_prices(ticker, start_date, end_date)
            if prices_df is None or prices_df.empty:
                raise ValueError(f"No price data available for {ticker}")
            
            # Create technical features
            technical_features = self._create_technical_features(prices_df)
            
            # Create fundamental features (will be empty if not available)
            fundamental_features = self._create_fundamental_features(ticker, start_date, end_date)
            
            # Create derived features
            derived_features = self._create_derived_features(prices_df, technical_features)
            
            # Combine features - only include non-empty DataFrames
            feature_list = [technical_features]
            
            # Only add fundamental features if they're not empty
            if not fundamental_features.empty:
                feature_list.append(fundamental_features)
                logger.info(f"Added fundamental features for {ticker}")
            else:
                logger.info(f"No fundamental features available for {ticker}, using technical features only")
            
            # Only add derived features if they're not empty
            if not derived_features.empty:
                feature_list.append(derived_features)
            
            # Combine all features
            all_features = pd.concat(feature_list, axis=1)
            
            # Create target variable
            target = self._create_target(prices_df, target_column, target_horizon)
            
            # Align features and target
            aligned_data = self._align_features_and_target(all_features, target)
            
            logger.info(f"Created {aligned_data[0].shape[1]} features for {ticker}")
            return aligned_data
            
        except Exception as e:
            logger.error(f"Feature creation failed for {ticker}: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Calculate all momentum indicators
            df_with_indicators = self.momentum_indicators.calculate_all_indicators(df)
            
            # Price-based features
            if self.feature_config['technical_features']['price_features']:
                features = self._add_price_features(features, df_with_indicators)
            
            # Volume-based features
            if self.feature_config['technical_features']['volume_features']:
                features = self._add_volume_features(features, df_with_indicators)
            
            # Momentum features
            if self.feature_config['technical_features']['momentum_features']:
                features = self._add_momentum_features(features, df_with_indicators)
            
            # Volatility features
            if self.feature_config['technical_features']['volatility_features']:
                features = self._add_volatility_features(features, df_with_indicators)
            
            # Trend features
            if self.feature_config['technical_features']['trend_features']:
                features = self._add_trend_features(features, df_with_indicators)
            
            return features
            
        except Exception as e:
            logger.error(f"Technical feature creation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        try:
            # Price ratios
            features['price_to_high'] = df['close_price'] / df['high_price']
            features['price_to_low'] = df['close_price'] / df['low_price']
            features['high_low_ratio'] = df['high_price'] / df['low_price']
            
            # Price changes
            for period in [1, 3, 5, 10, 20]:
                features[f'price_change_{period}d'] = df['close_price'].pct_change(period)
                features[f'price_change_abs_{period}d'] = df['close_price'].pct_change(period).abs()
            
            # Price levels
            features['price_position'] = (df['close_price'] - df['low_price']) / (df['high_price'] - df['low_price'])
            
            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                if f'sma_{period}' in df.columns:
                    features[f'price_to_sma_{period}'] = df['close_price'] / df[f'sma_{period}']
                    features[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5)
            
            return features
            
        except Exception as e:
            logger.error(f"Price feature creation failed: {e}")
            return features
    
    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        try:
            # Volume ratios
            features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(5).mean()
            
            # Volume-price relationship
            features['volume_price_trend'] = (df['volume'] * df['close_price']).rolling(10).sum()
            features['volume_weighted_price'] = (df['volume'] * df['close_price']).rolling(10).sum() / df['volume'].rolling(10).sum()
            
            # Volume momentum
            features['volume_momentum'] = df['volume'].pct_change(5)
            features['volume_acceleration'] = df['volume'].pct_change(5).diff(3)
            
            return features
            
        except Exception as e:
            logger.error(f"Volume feature creation failed: {e}")
            return features
    
    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        try:
            # RSI features
            if 'rsi_14' in df.columns:
                features['rsi_momentum'] = df['rsi_14'].diff(3)
                features['rsi_ma_ratio'] = df['rsi_14'] / df['rsi_14'].rolling(10).mean()
                features['rsi_extreme'] = np.where(df['rsi_14'] > 70, 1, np.where(df['rsi_14'] < 30, -1, 0))
            
            # MACD features
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                features['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
                features['macd_histogram_momentum'] = df['macd_histogram'].diff(3)
            
            # Stochastic features
            if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
                features['stoch_cross'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
                features['stoch_momentum'] = df['stoch_k'].diff(5)
            
            return features
            
        except Exception as e:
            logger.error(f"Momentum feature creation failed: {e}")
            return features
    
    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        try:
            # ATR features
            if 'atr_14' in df.columns:
                features['atr_ratio'] = df['atr_14'] / df['close_price']
                features['atr_momentum'] = df['atr_14'].pct_change(5)
                features['atr_ma_ratio'] = df['atr_14'] / df['atr_14'].rolling(20).mean()
            
            # Bollinger Bands features
            if all(col in df.columns for col in ['bollinger_upper', 'bollinger_lower']):
                features['bb_position'] = (df['close_price'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
                features['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['close_price']
            
            # Historical volatility
            for period in [5, 10, 20, 50]:
                features[f'volatility_{period}d'] = df['close_price'].pct_change().rolling(period).std() * np.sqrt(252)
            
            return features
            
        except Exception as e:
            logger.error(f"Volatility feature creation failed: {e}")
            return features
    
    def _add_trend_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features."""
        try:
            # Moving average trends
            for period in [20, 50, 200]:
                if f'sma_{period}' in df.columns:
                    features[f'sma_{period}_trend'] = np.where(df[f'sma_{period}'] > df[f'sma_{period}'].shift(5), 1, -1)
                    features[f'sma_{period}_strength'] = (df['close_price'] - df[f'sma_{period}']) / df[f'sma_{period}']
            
            # ADX features
            if 'adx' in df.columns:
                features['adx_trend_strength'] = np.where(df['adx'] > 25, 1, 0)
                features['adx_momentum'] = df['adx'].diff(5)
            
            return features
            
        except Exception as e:
            logger.error(f"Trend feature creation failed: {e}")
            return features
    
    def _create_fundamental_features(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create fundamental analysis features."""
        try:
            features = pd.DataFrame()
            
            # Get fundamental data
            fundamental_data = get_financial_metrics(ticker, end_date)
            if fundamental_data is None or len(fundamental_data) == 0:
                logger.debug(f"No fundamental data available for {ticker} - skipping fundamental features")
                return features
            
            # Valuation features
            if self.feature_config['fundamental_features']['valuation_features']:
                features = self._add_valuation_features(features, fundamental_data)
            
            # Financial features
            if self.feature_config['fundamental_features']['financial_features']:
                features = self._add_financial_features(features, fundamental_data)
            
            # Check if we actually got any features
            if features.empty:
                logger.debug(f"Fundamental data available but no features created for {ticker}")
            else:
                logger.debug(f"Created {features.shape[1]} fundamental features for {ticker}")
            
            return features
            
        except Exception as e:
            logger.debug(f"Fundamental feature creation failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _add_valuation_features(self, features: pd.DataFrame, data: Any) -> pd.DataFrame:
        """Add valuation-based features."""
        try:
            # Basic valuation ratios
            if hasattr(data, 'price_to_earnings_ratio') and data.price_to_earnings_ratio:
                features['pe_ratio'] = data.price_to_earnings_ratio
            
            if hasattr(data, 'price_to_book_ratio') and data.price_to_book_ratio:
                features['pb_ratio'] = data.price_to_book_ratio
            
            if hasattr(data, 'price_to_sales_ratio') and data.price_to_sales_ratio:
                features['ps_ratio'] = data.price_to_sales_ratio
            
            return features
            
        except Exception as e:
            logger.error(f"Valuation feature creation failed: {e}")
            return features
    
    def _add_financial_features(self, features: pd.DataFrame, data: Any) -> pd.DataFrame:
        """Add financial performance features."""
        try:
            # Profitability ratios
            if hasattr(data, 'return_on_equity') and data.return_on_equity:
                features['roe'] = data.return_on_equity
            
            if hasattr(data, 'return_on_assets') and data.return_on_assets:
                features['roa'] = data.return_on_assets
            
            return features
            
        except Exception as e:
            logger.error(f"Financial feature creation failed: {e}")
            return features
    
    def _create_derived_features(self, df: pd.DataFrame, technical_features: pd.DataFrame) -> pd.DataFrame:
        """Create derived and interaction features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            if self.feature_config['derived_features']['interaction_features']:
                features = self._add_interaction_features(features, technical_features)
            
            if self.feature_config['derived_features']['lag_features']:
                features = self._add_lag_features(features, technical_features)
            
            if self.feature_config['derived_features']['rolling_features']:
                features = self._add_rolling_features(features, technical_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Derived feature creation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _add_interaction_features(self, features: pd.DataFrame, technical_features: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different indicators."""
        try:
            # RSI and volume interaction
            if 'rsi_14' in technical_features.columns and 'volume_sma_ratio' in technical_features.columns:
                features['rsi_volume_interaction'] = technical_features['rsi_14'] * technical_features['volume_sma_ratio']
            
            # MACD and volatility interaction
            if 'macd' in technical_features.columns and 'volatility_20d' in technical_features.columns:
                features['macd_volatility_interaction'] = technical_features['macd'] * technical_features['volatility_20d']
            
            return features
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {e}")
            return features
    
    def _add_lag_features(self, features: pd.DataFrame, technical_features: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        try:
            # Lag important features
            important_features = ['rsi_14', 'macd', 'volume_sma_ratio', 'price_change_5d']
            
            for feature in important_features:
                if feature in technical_features.columns:
                    for lag in [1, 3, 5, 10]:
                        features[f'{feature}_lag_{lag}'] = technical_features[feature].shift(lag)
            
            return features
            
        except Exception as e:
            logger.error(f"Lag feature creation failed: {e}")
            return features
    
    def _add_rolling_features(self, features: pd.DataFrame, technical_features: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features."""
        try:
            # Rolling statistics for important features
            important_features = ['rsi_14', 'macd', 'volume_sma_ratio']
            
            for feature in important_features:
                if feature in technical_features.columns:
                    for window in [5, 10, 20]:
                        features[f'{feature}_rolling_mean_{window}'] = technical_features[feature].rolling(window).mean()
                        features[f'{feature}_rolling_std_{window}'] = technical_features[feature].rolling(window).std()
            
            return features
            
        except Exception as e:
            logger.error(f"Rolling feature creation failed: {e}")
            return features
    
    def _create_target(self, df: pd.DataFrame, target_column: str, target_horizon: int) -> pd.Series:
        """Create target variable for ML modeling."""
        try:
            if target_column == 'close_price':
                # Future price change
                target = df['close_price'].shift(-target_horizon) / df['close_price'] - 1
                target.name = f'future_return_{target_horizon}d'
            else:
                target = df[target_column].shift(-target_horizon)
                target.name = f'future_{target_column}_{target_horizon}d'
            
            return target
            
        except Exception as e:
            logger.error(f"Target creation failed: {e}")
            return pd.Series(index=df.index)
    
    def _align_features_and_target(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target, removing NaN values."""
        try:
            # Combine features and target
            combined = pd.concat([features, target], axis=1)
            
            # Remove rows with NaN values
            combined_clean = combined.dropna()
            
            # Separate features and target
            target_clean = combined_clean[target.name]
            features_clean = combined_clean.drop(columns=[target.name])
            
            return features_clean, target_clean
            
        except Exception as e:
            logger.error(f"Feature-target alignment failed: {e}")
            return pd.DataFrame(), pd.Series()
    
    def fit_transform(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit scaler and transform features."""
        try:
            # Store feature names
            self.feature_names = features.columns.tolist()
            
            # Fit and transform scaler
            features_scaled = self.scaler.fit_transform(features)
            self.scaler_fitted = True
            
            # Convert back to DataFrame
            features_scaled_df = pd.DataFrame(features_scaled, 
                                            index=features.index, 
                                            columns=features.columns)
            
            logger.info(f"Features scaled using {self.scaler_type} scaler")
            return features_scaled_df, target
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return features, target
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        try:
            if not self.scaler_fitted:
                raise ValueError("Scaler must be fitted before transforming")
            
            features_scaled = self.scaler.transform(features)
            features_scaled_df = pd.DataFrame(features_scaled, 
                                            index=features.index, 
                                            columns=features.columns)
            
            return features_scaled_df
            
        except Exception as e:
            logger.error(f"Feature transformation failed: {e}")
            return features
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, model.feature_importances_))
                self.feature_importance = importance_dict
                return importance_dict
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return {}
                
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering summary."""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'scaler_type': self.scaler_type,
            'scaler_fitted': self.scaler_fitted,
            'feature_config': self.feature_config
        } 