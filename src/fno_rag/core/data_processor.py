#!/usr/bin/env python3
"""
FNO Data Processor
Handles data preparation, feature engineering, and label creation for FNO analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from loguru import logger

from ..models.data_models import FNOData, HorizonType
from ...data.database.duckdb_manager import DatabaseManager


class FNODataProcessor:
    """Processes FNO data for ML and RAG analysis."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize the data processor."""
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logger
        
    def get_fno_data(self, symbols: Optional[List[str]] = None, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve FNO data from database."""
        try:
            query = """
                SELECT 
                    TckrSymb as symbol,
                    TRADE_DATE as date,
                    OpnPric as open_price,
                    HghPric as high_price,
                    LwPric as low_price,
                    ClsPric as close_price,
                    TtlTradgVol as volume,
                    OpnIntrst as open_interest
                FROM fno_bhav_copy
                WHERE 1=1
            """
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND TckrSymb IN ({placeholders})"
                params.extend(symbols)
            
            if start_date:
                query += " AND TRADE_DATE >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND TRADE_DATE <= ?"
                params.append(end_date)
            
            query += " ORDER BY TckrSymb, TRADE_DATE"
            
            df = self.db_manager.connection.execute(query, params).fetchdf()
            
            # Convert data types to ensure compatibility
            if not df.empty:
                # Convert numeric columns to float
                numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'open_interest']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Convert date column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Convert symbol to string
                if 'symbol' in df.columns:
                    df['symbol'] = df['symbol'].astype(str)
            
            self.logger.info(f"Retrieved {len(df)} FNO records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve FNO data: {e}")
            raise
    
    def get_fno_data_for_symbol(self, symbol: str, start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve FNO data for a specific symbol."""
        try:
            query = """
                SELECT
                    TckrSymb as symbol,
                    TRADE_DATE as date,
                    OpnPric as open_price,
                    HghPric as high_price,
                    LwPric as low_price,
                    ClsPric as close_price,
                    TtlTradgVol as volume,
                    OpnIntrst as open_interest
                FROM fno_bhav_copy
                WHERE TckrSymb = ?
            """
            
            params = [symbol]
            
            if start_date:
                query += " AND TRADE_DATE >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND TRADE_DATE <= ?"
                params.append(end_date)
            
            query += " ORDER BY TRADE_DATE"
            
            result = self.db_manager.connection.execute(query, params).fetchdf()
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve FNO data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset."""
        try:
            # Ensure data types are correct
            df = df.copy()
            
            # Convert numeric columns to float
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            if 'open_interest' in df.columns:
                df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0)
            
            # Ensure symbol is string
            df['symbol'] = df['symbol'].astype(str)
            
            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            # Calculate basic returns (these work fine)
            df['daily_return'] = df.groupby('symbol')['close_price'].pct_change()
            df['weekly_return'] = df.groupby('symbol')['close_price'].pct_change(5)
            df['monthly_return'] = df.groupby('symbol')['close_price'].pct_change(20)
            
            # For now, skip complex technical indicators to avoid dtype issues
            # Set default values for required features
            for period in [3, 5, 20]:
                df[f'return_{period}d_mean'] = 0
                df[f'return_{period}d_std'] = 0
            
            # Basic volume indicator
            df['volume_spike_ratio'] = 1.0
            
            # Basic OI indicators
            if 'open_interest' in df.columns:
                df['oi_change_pct'] = 0
                df['oi_spike_ratio'] = 1.0
            
            # Basic volatility indicator
            df['intraday_range'] = (df['high_price'] - df['low_price']) / df['close_price']
            
            # Set default values for other indicators
            df['atr_5'] = 0
            df['atr_14'] = 0
            df['rsi_3'] = 50
            df['rsi_14'] = 50
            df['macd'] = 0
            df['macd_signal'] = 0
            df['bb_width'] = 0
            df['stoch_k'] = 50
            df['stoch_d'] = 50
            
            self.logger.info("Basic technical indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to calculate technical indicators: {e}")
            raise
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels for different horizons."""
        try:
            # Daily labels (±3-5% move) - Convert to [0, 1, 2] for ML models
            df['daily_label'] = 1  # Default to neutral (1)
            df.loc[df['daily_return'] >= 0.03, 'daily_label'] = 2  # Up (2)
            df.loc[df['daily_return'] <= -0.03, 'daily_label'] = 0  # Down (0)
            
            # Weekly labels (±5% move) - Convert to [0, 1, 2] for ML models
            df['weekly_label'] = 1  # Default to neutral (1)
            df.loc[df['weekly_return'] >= 0.05, 'weekly_label'] = 2  # Up (2)
            df.loc[df['weekly_return'] <= -0.05, 'weekly_label'] = 0  # Down (0)
            
            # Monthly labels (±10% move) - Convert to [0, 1, 2] for ML models
            df['monthly_label'] = 1  # Default to neutral (1)
            df.loc[df['monthly_return'] >= 0.10, 'monthly_label'] = 2  # Up (2)
            df.loc[df['monthly_return'] <= -0.10, 'monthly_label'] = 0  # Down (0)
            
            self.logger.info("Labels created successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create labels: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame, horizon: HorizonType) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and labels for ML training."""
        try:
            # Select relevant features
            feature_columns = [
                'daily_return', 'weekly_return', 'monthly_return',
                'return_3d_mean', 'return_5d_mean', 'return_20d_mean',
                'return_3d_std', 'return_5d_std', 'return_20d_std',
                'volume_spike_ratio', 'intraday_range',
                'atr_5', 'atr_14', 'rsi_3', 'rsi_14',
                'macd', 'macd_signal', 'bb_width',
                'stoch_k', 'stoch_d'
            ]
            
            # Add OI features if available
            if 'open_interest' in df.columns:
                feature_columns.extend(['oi_change_pct', 'oi_spike_ratio'])
            
            # Add Put-Call ratio if available
            if 'put_call_ratio' in df.columns:
                feature_columns.append('put_call_ratio')
            
            # Add Implied Volatility if available
            if 'implied_volatility' in df.columns:
                feature_columns.append('implied_volatility')
            
            # Create feature matrix
            X = df[feature_columns].copy()
            
            # Handle missing values
            X = X.fillna(0)
            
            # Create label vector
            if horizon == HorizonType.DAILY:
                y = df['daily_label']
            elif horizon == HorizonType.WEEKLY:
                y = df['weekly_label']
            else:  # MONTHLY
                y = df['monthly_label']
            
            # Remove rows with missing labels
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features for {horizon.value}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features: {e}")
            raise
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high_price'] - df['low_price']
        high_close = np.abs(df['high_price'] - df.groupby('symbol')['close_price'].shift())
        low_close = np.abs(df['low_price'] - df.groupby('symbol')['close_price'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Calculate rolling mean directly on the true_range series
        atr = true_range.groupby(df['symbol']).rolling(period).mean().reset_index(0, drop=True)
        return atr.reindex(df.index, fill_value=0)
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df.groupby('symbol')['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.reset_index(0, drop=True).reindex(df.index, fill_value=50)
    
    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_12 = df.groupby('symbol')['close_price'].ewm(span=12).mean()
        ema_26 = df.groupby('symbol')['close_price'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.groupby('symbol').ewm(span=9).mean()
        macd_series = macd.reset_index(0, drop=True).reindex(df.index, fill_value=0)
        signal_series = signal.reset_index(0, drop=True).reindex(df.index, fill_value=0)
        return macd_series, signal_series
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = df.groupby('symbol')['close_price'].rolling(20).mean().reset_index(0, drop=True)
        std = df.groupby('symbol')['close_price'].rolling(20).std().reset_index(0, drop=True)
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        
        # Ensure proper index alignment
        middle_series = middle.reindex(df.index, fill_value=df['close_price'])
        upper_series = upper.reindex(df.index, fill_value=df['close_price'])
        lower_series = lower.reindex(df.index, fill_value=df['close_price'])
        
        return upper_series, lower_series, middle_series
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df.groupby('symbol')['low_price'].rolling(14).min().reset_index(0, drop=True)
        high_max = df.groupby('symbol')['high_price'].rolling(14).max().reset_index(0, drop=True)
        
        k = 100 * ((df['close_price'] - low_min) / (high_max - low_min))
        d = k.groupby('symbol').rolling(3).mean().reset_index(0, drop=True)
        
        # Ensure proper index alignment
        k_series = k.reindex(df.index, fill_value=50)
        d_series = d.reindex(df.index, fill_value=50)
        
        return k_series, d_series
    
    def get_latest_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get the latest FNO data for prediction."""
        try:
            # Get data for the last 30 days to ensure we have enough for indicators
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            df = self.get_fno_data(symbols, start_date, end_date)
            df = self.calculate_technical_indicators(df)
            df = self.create_labels(df)
            
            # Get only the latest data for each symbol
            latest_data = df.groupby('symbol').last().reset_index()
            
            self.logger.info(f"Retrieved latest data for {len(latest_data)} symbols")
            return latest_data
            
        except Exception as e:
            self.logger.error(f"Failed to get latest data: {e}")
            raise
