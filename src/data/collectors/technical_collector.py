"""
Technical Data Collector for AI Hedge Fund
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import yfinance as yf
from loguru import logger

from ..database.models import TechnicalData
from ...tools.enhanced_api import get_prices

class TechnicalDataCollector:
    """Collects technical data including price data and indicators."""
    
    def __init__(self):
        """Initialize technical data collector."""
        self.indicators = {
            'sma_20': self._calculate_sma_20,
            'sma_50': self._calculate_sma_50,
            'sma_200': self._calculate_sma_200,
            'rsi_14': self._calculate_rsi_14,
            'macd': self._calculate_macd,
            'bollinger_bands': self._calculate_bollinger_bands,
            'atr_14': self._calculate_atr_14
        }
    
    def collect_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Collect technical data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with technical data
        """
        try:
            logger.info(f"Collecting technical data for {ticker} from {start_date} to {end_date}")
            
            # Get price data
            price_data = self._get_price_data(ticker, start_date, end_date)
            if price_data is None or price_data.empty:
                logger.warning(f"No price data found for {ticker}")
                return None
            
            # Calculate technical indicators
            technical_data = self._calculate_indicators(price_data)
            
            # Add metadata
            technical_data['ticker'] = ticker
            technical_data['created_at'] = datetime.now()
            technical_data['updated_at'] = datetime.now()
            
            logger.info(f"Collected {len(technical_data)} technical data records for {ticker}")
            return technical_data
            
        except Exception as e:
            logger.error(f"Failed to collect technical data for {ticker}: {e}")
            return None
    
    def _get_price_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get price data from multiple sources."""
        try:
            # Try enhanced API first
            prices = get_prices(ticker, start_date, end_date)
            if prices and len(prices) > 0:
                # Convert to DataFrame
                df = pd.DataFrame([price.dict() for price in prices])
                df['trade_date'] = pd.to_datetime(df['date'])
                df = df.sort_values('trade_date')
                return df
            
            # Fallback to yfinance
            logger.info(f"Falling back to yfinance for {ticker}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return None
            
            # Standardize column names
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            # Rename columns to match our schema
            column_mapping = {
                'date': 'trade_date',
                'open': 'open_price',
                'high': 'high_price',
                'low': 'low_price',
                'close': 'close_price',
                'volume': 'volume',
                'adj close': 'adjusted_close'
            }
            
            df = df.rename(columns=column_mapping)
            df = df[list(column_mapping.values())]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get price data for {ticker}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        try:
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # Calculate each indicator
            for indicator_name, indicator_func in self.indicators.items():
                try:
                    if indicator_name == 'macd':
                        # MACD returns multiple columns
                        macd_data = indicator_func(result_df)
                        result_df = pd.concat([result_df, macd_data], axis=1)
                    elif indicator_name == 'bollinger_bands':
                        # Bollinger bands returns multiple columns
                        bb_data = indicator_func(result_df)
                        result_df = pd.concat([result_df, bb_data], axis=1)
                    else:
                        # Single column indicators
                        result_df[indicator_name] = indicator_func(result_df)
                except Exception as e:
                    logger.warning(f"Failed to calculate {indicator_name}: {e}")
                    result_df[indicator_name] = np.nan
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return df
    
    def _calculate_sma_20(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 20-day Simple Moving Average."""
        return df['close_price'].rolling(window=20).mean()
    
    def _calculate_sma_50(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 50-day Simple Moving Average."""
        return df['close_price'].rolling(window=50).mean()
    
    def _calculate_sma_200(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 200-day Simple Moving Average."""
        return df['close_price'].rolling(window=200).mean()
    
    def _calculate_rsi_14(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 14-day Relative Strength Index."""
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        exp1 = df['close_price'].ewm(span=12, adjust=False).mean()
        exp2 = df['close_price'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram
        })
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma_20 = df['close_price'].rolling(window=20).mean()
        std_20 = df['close_price'].rolling(window=20).std()
        
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        return pd.DataFrame({
            'bollinger_upper': upper_band,
            'bollinger_lower': lower_band,
            'bollinger_middle': sma_20
        })
    
    def _calculate_atr_14(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 14-day Average True Range."""
        high_low = df['high_price'] - df['low_price']
        high_close = np.abs(df['high_price'] - df['close_price'].shift())
        low_close = np.abs(df['low_price'] - df['close_price'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=14).mean()
        
        return atr
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate technical data quality."""
        validation_result = {
            'total_records': len(df),
            'missing_values': {},
            'data_quality_score': 0.0,
            'issues': []
        }
        
        # Check for missing values
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                validation_result['missing_values'][column] = missing_count
        
        # Check for negative prices
        price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
        for col in price_columns:
            if col in df.columns:
                negative_prices = (df[col] <= 0).sum()
                if negative_prices > 0:
                    validation_result['issues'].append(f"Found {negative_prices} negative prices in {col}")
        
        # Check for volume issues
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                validation_result['issues'].append(f"Found {negative_volume} negative volumes")
        
        # Calculate data quality score
        total_cells = len(df) * len(df.columns)
        missing_cells = sum(validation_result['missing_values'].values())
        validation_result['data_quality_score'] = ((total_cells - missing_cells) / total_cells) * 100
        
        return validation_result
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for technical data."""
        summary = {
            'date_range': {
                'start': df['trade_date'].min().strftime('%Y-%m-%d') if not df.empty else None,
                'end': df['trade_date'].max().strftime('%Y-%m-%d') if not df.empty else None,
                'total_days': len(df)
            },
            'price_statistics': {},
            'volume_statistics': {},
            'indicator_coverage': {}
        }
        
        if not df.empty:
            # Price statistics
            price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
            for col in price_columns:
                if col in df.columns:
                    summary['price_statistics'][col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
            
            # Volume statistics
            if 'volume' in df.columns:
                summary['volume_statistics'] = {
                    'min': df['volume'].min(),
                    'max': df['volume'].max(),
                    'mean': df['volume'].mean(),
                    'std': df['volume'].std()
                }
            
            # Indicator coverage
            indicator_columns = ['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd', 'bollinger_upper']
            for col in indicator_columns:
                if col in df.columns:
                    coverage = (df[col].notna().sum() / len(df)) * 100
                    summary['indicator_coverage'][col] = f"{coverage:.1f}%"
        
        return summary 