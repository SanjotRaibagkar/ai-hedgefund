"""
Intraday Feature Engineer
Creates features for intraday ML prediction from collected data.
"""

import sys
import os
sys.path.append('./src')

import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import talib

from .data_collector import IntradayDataCollector


class IntradayFeatureEngineer:
    """Creates features for intraday ML prediction."""
    
    def __init__(self, db_path: str = "data/intraday_ml_data.duckdb"):
        """
        Initialize the feature engineer.
        
        Args:
            db_path: Path to DuckDB database for intraday ML data
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        logger.info("🚀 Intraday Feature Engineer initialized")
    
    def get_options_features(self, index_symbol: str, timestamp: datetime) -> Dict[str, float]:
        """
        Extract options chain features for a specific timestamp.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            timestamp: Timestamp for feature extraction
            
        Returns:
            Dictionary with options features
        """
        try:
            # Get options data for the timestamp
            query = """
                SELECT * FROM intraday_options_data 
                WHERE index_symbol = ? 
                AND timestamp = ?
                ORDER BY strike_price, option_type
            """
            
            options_data = self.conn.execute(query, [index_symbol, timestamp]).fetchdf()
            
            if options_data.empty:
                logger.warning(f"⚠️ No options data found for {index_symbol} at {timestamp}")
                return {}
            
            features = {}
            
            # Get ATM options data
            atm_options = self._get_atm_options_data(options_data)
            if not atm_options.empty:
                features.update(self._extract_atm_features(atm_options))
            
            # Get OI change features
            features.update(self._extract_oi_features(options_data))
            
            # Get PCR features
            features.update(self._extract_pcr_features(options_data))
            
            # Get IV features
            features.update(self._extract_iv_features(options_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting options features: {e}")
            return {}
    
    def _get_atm_options_data(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Get ATM (At-The-Money) options data."""
        try:
            # Get spot price from the data
            spot_price = options_data['spot_price'].iloc[0]
            atm_strike = options_data['atm_strike'].iloc[0]
            
            # Filter ATM options
            atm_options = options_data[options_data['strike_price'] == atm_strike]
            
            return atm_options
            
        except Exception as e:
            logger.error(f"❌ Error getting ATM options data: {e}")
            return pd.DataFrame()
    
    def _extract_atm_features(self, atm_options: pd.DataFrame) -> Dict[str, float]:
        """Extract features from ATM options."""
        try:
            features = {}
            
            # Separate CE and PE options
            ce_options = atm_options[atm_options['option_type'] == 'CE']
            pe_options = atm_options[atm_options['option_type'] == 'PE']
            
            # CE features
            if not ce_options.empty:
                ce_data = ce_options.iloc[0]
                features.update({
                    'atm_ce_delta': ce_data.get('delta', 0),
                    'atm_ce_theta': ce_data.get('theta', 0),
                    'atm_ce_vega': ce_data.get('vega', 0),
                    'atm_ce_gamma': ce_data.get('gamma', 0),
                    'atm_ce_iv': ce_data.get('implied_volatility', 0),
                    'atm_ce_premium': ce_data.get('last_price', 0),
                    'atm_ce_oi': ce_data.get('open_interest', 0),
                    'atm_ce_volume': ce_data.get('volume', 0)
                })
            else:
                features.update({
                    'atm_ce_delta': 0, 'atm_ce_theta': 0, 'atm_ce_vega': 0,
                    'atm_ce_gamma': 0, 'atm_ce_iv': 0, 'atm_ce_premium': 0,
                    'atm_ce_oi': 0, 'atm_ce_volume': 0
                })
            
            # PE features
            if not pe_options.empty:
                pe_data = pe_options.iloc[0]
                features.update({
                    'atm_pe_delta': pe_data.get('delta', 0),
                    'atm_pe_theta': pe_data.get('theta', 0),
                    'atm_pe_vega': pe_data.get('vega', 0),
                    'atm_pe_gamma': pe_data.get('gamma', 0),
                    'atm_pe_iv': pe_data.get('implied_volatility', 0),
                    'atm_pe_premium': pe_data.get('last_price', 0),
                    'atm_pe_oi': pe_data.get('open_interest', 0),
                    'atm_pe_volume': pe_data.get('volume', 0)
                })
            else:
                features.update({
                    'atm_pe_delta': 0, 'atm_pe_theta': 0, 'atm_pe_vega': 0,
                    'atm_pe_gamma': 0, 'atm_pe_iv': 0, 'atm_pe_premium': 0,
                    'atm_pe_oi': 0, 'atm_pe_volume': 0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting ATM features: {e}")
            return {}
    
    def _extract_oi_features(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Extract Open Interest change features."""
        try:
            features = {}
            
            # Get OI changes for different strikes
            strikes = sorted(options_data['strike_price'].unique())
            spot_price = options_data['spot_price'].iloc[0]
            
            # Find ATM and nearby strikes
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            atm_index = strikes.index(atm_strike)
            
            # ATM OI changes
            atm_ce_oi_change = options_data[
                (options_data['strike_price'] == atm_strike) & 
                (options_data['option_type'] == 'CE')
            ]['change_in_oi'].sum()
            
            atm_pe_oi_change = options_data[
                (options_data['strike_price'] == atm_strike) & 
                (options_data['option_type'] == 'PE')
            ]['change_in_oi'].sum()
            
            features.update({
                'atm_ce_oi_change': atm_ce_oi_change,
                'atm_pe_oi_change': atm_pe_oi_change,
                'atm_oi_change_ratio': atm_pe_oi_change / (atm_ce_oi_change + 1e-8)
            })
            
            # OTM OI changes (1 strike away)
            if atm_index + 1 < len(strikes):
                otm_strike = strikes[atm_index + 1]
                otm_ce_oi_change = options_data[
                    (options_data['strike_price'] == otm_strike) & 
                    (options_data['option_type'] == 'CE')
                ]['change_in_oi'].sum()
                
                otm_pe_oi_change = options_data[
                    (options_data['strike_price'] == otm_strike) & 
                    (options_data['option_type'] == 'PE')
                ]['change_in_oi'].sum()
                
                features.update({
                    'otm_ce_oi_change': otm_ce_oi_change,
                    'otm_pe_oi_change': otm_pe_oi_change,
                    'otm_oi_change_ratio': otm_pe_oi_change / (otm_ce_oi_change + 1e-8)
                })
            else:
                features.update({
                    'otm_ce_oi_change': 0,
                    'otm_pe_oi_change': 0,
                    'otm_oi_change_ratio': 0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting OI features: {e}")
            return {}
    
    def _extract_pcr_features(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Extract Put-Call Ratio features."""
        try:
            features = {}
            
            # Overall PCR (OI)
            total_ce_oi = options_data[options_data['option_type'] == 'CE']['open_interest'].sum()
            total_pe_oi = options_data[options_data['option_type'] == 'PE']['open_interest'].sum()
            
            pcr_oi = total_pe_oi / (total_ce_oi + 1e-8)
            
            # Overall PCR (Volume)
            total_ce_volume = options_data[options_data['option_type'] == 'CE']['volume'].sum()
            total_pe_volume = options_data[options_data['option_type'] == 'PE']['volume'].sum()
            
            pcr_volume = total_pe_volume / (total_ce_volume + 1e-8)
            
            features.update({
                'pcr_oi': pcr_oi,
                'pcr_volume': pcr_volume,
                'pcr_ratio': pcr_oi / (pcr_volume + 1e-8)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting PCR features: {e}")
            return {}
    
    def _extract_iv_features(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Extract Implied Volatility features."""
        try:
            features = {}
            
            # ATM IV features
            atm_strike = options_data['atm_strike'].iloc[0]
            atm_options = options_data[options_data['strike_price'] == atm_strike]
            
            atm_ce_iv = atm_options[atm_options['option_type'] == 'CE']['implied_volatility'].mean()
            atm_pe_iv = atm_options[atm_options['option_type'] == 'PE']['implied_volatility'].mean()
            
            features.update({
                'atm_ce_iv': atm_ce_iv,
                'atm_pe_iv': atm_pe_iv,
                'atm_iv_skew': atm_pe_iv - atm_ce_iv,
                'atm_iv_ratio': atm_pe_iv / (atm_ce_iv + 1e-8)
            })
            
            # IV change features (if we have historical data)
            # This would require comparing with previous timestamps
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting IV features: {e}")
            return {}
    
    def get_index_features(self, index_symbol: str, timestamp: datetime) -> Dict[str, float]:
        """
        Extract index technical features for a specific timestamp.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            timestamp: Timestamp for feature extraction
            
        Returns:
            Dictionary with index features
        """
        try:
            # Get index data for the timestamp and recent history
            query = """
                SELECT * FROM intraday_index_data 
                WHERE index_symbol = ? 
                AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 100
            """
            
            index_data = self.conn.execute(query, [index_symbol, timestamp]).fetchdf()
            
            if index_data.empty:
                logger.warning(f"⚠️ No index data found for {index_symbol} at {timestamp}")
                return {}
            
            # Sort by timestamp (ascending for technical indicators)
            index_data = index_data.sort_values('timestamp')
            
            features = {}
            
            # Basic OHLCV features
            if not index_data.empty:
                latest_data = index_data.iloc[-1]
                features.update({
                    'open_price': latest_data['open_price'],
                    'high_price': latest_data['high_price'],
                    'low_price': latest_data['low_price'],
                    'close_price': latest_data['close_price'],
                    'volume': latest_data['volume'],
                    'turnover': latest_data['turnover']
                })
            
            # Technical indicators
            if len(index_data) >= 14:  # Minimum data for indicators
                features.update(self._calculate_technical_indicators(index_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting index features: {e}")
            return {}
    
    def _calculate_technical_indicators(self, index_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from index data."""
        try:
            features = {}
            
            # Convert to numpy arrays for talib
            close_prices = index_data['close_price'].values
            high_prices = index_data['high_price'].values
            low_prices = index_data['low_price'].values
            volumes = index_data['volume'].values
            
            # RSI (14 periods)
            if len(close_prices) >= 14:
                rsi = talib.RSI(close_prices, timeperiod=14)
                features['rsi_14'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            # MACD (12, 26, 9)
            if len(close_prices) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                features['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
                features['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
                features['macd_histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            # Bollinger Bands (20, 2)
            if len(close_prices) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
                current_price = close_prices[-1]
                
                features['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price
                features['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else current_price
                features['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price
                
                # BB deviation
                if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
                    bb_range = bb_upper[-1] - bb_lower[-1]
                    if bb_range > 0:
                        bb_deviation = (current_price - bb_lower[-1]) / bb_range
                        features['bb_deviation'] = bb_deviation
                    else:
                        features['bb_deviation'] = 0.5
                else:
                    features['bb_deviation'] = 0.5
            
            # VWAP (Volume Weighted Average Price)
            if len(index_data) > 0:
                vwap = np.sum(close_prices * volumes) / np.sum(volumes)
                features['vwap'] = vwap
                features['vwap_deviation'] = (close_prices[-1] - vwap) / vwap if vwap > 0 else 0
            
            # 15-minute returns
            if len(close_prices) >= 2:
                features['return_15min'] = (close_prices[-1] - close_prices[-2]) / close_prices[-2] if close_prices[-2] > 0 else 0
            
            # Price momentum
            if len(close_prices) >= 5:
                features['momentum_5'] = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if close_prices[-5] > 0 else 0
            
            if len(close_prices) >= 10:
                features['momentum_10'] = (close_prices[-1] - close_prices[-10]) / close_prices[-10] if close_prices[-10] > 0 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}
    
    def get_sentiment_features(self, timestamp: datetime) -> Dict[str, float]:
        """
        Extract market sentiment features for a specific timestamp.
        
        Args:
            timestamp: Timestamp for feature extraction
            
        Returns:
            Dictionary with sentiment features
        """
        try:
            features = {}
            
            # Get FII/DII data for the current date
            current_date = timestamp.date()
            query = """
                SELECT * FROM intraday_fii_dii_data 
                WHERE date = ?
            """
            
            fii_dii_data = self.conn.execute(query, [current_date]).fetchdf()
            
            if not fii_dii_data.empty:
                fii_data = fii_dii_data.iloc[0]
                features.update({
                    'fii_buy': fii_data['fii_buy'],
                    'fii_sell': fii_data['fii_sell'],
                    'fii_net': fii_data['fii_net'],
                    'dii_buy': fii_data['dii_buy'],
                    'dii_sell': fii_data['dii_sell'],
                    'dii_net': fii_data['dii_net'],
                    'fii_dii_ratio': fii_data['fii_net'] / (fii_data['dii_net'] + 1e-8)
                })
            else:
                # Use zeros if no data available
                features.update({
                    'fii_buy': 0, 'fii_sell': 0, 'fii_net': 0,
                    'dii_buy': 0, 'dii_sell': 0, 'dii_net': 0,
                    'fii_dii_ratio': 0
                })
            
            # Get VIX data
            vix_query = """
                SELECT * FROM intraday_vix_data 
                WHERE timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            vix_data = self.conn.execute(vix_query, [timestamp]).fetchdf()
            
            if not vix_data.empty:
                vix_row = vix_data.iloc[0]
                features.update({
                    'vix_value': vix_row['vix_value'],
                    'vix_change': vix_row['vix_change']
                })
            else:
                features.update({
                    'vix_value': 0,
                    'vix_change': 0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting sentiment features: {e}")
            return {}
    
    def create_labels(self, index_symbol: str, timestamp: datetime, future_minutes: int = 15) -> Dict[str, Any]:
        """
        Create labels for the target variable (15-minute future direction).
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            timestamp: Current timestamp
            future_minutes: Minutes into the future to predict
            
        Returns:
            Dictionary with label information
        """
        try:
            # Get current close price
            current_query = """
                SELECT close_price FROM intraday_index_data 
                WHERE index_symbol = ? AND timestamp = ?
            """
            
            current_result = self.conn.execute(current_query, [index_symbol, timestamp]).fetchdf()
            
            if current_result.empty:
                logger.warning(f"⚠️ No current price data for {index_symbol} at {timestamp}")
                return {}
            
            current_close = current_result.iloc[0]['close_price']
            
            # Get future close price
            future_timestamp = timestamp + timedelta(minutes=future_minutes)
            future_query = """
                SELECT close_price FROM intraday_index_data 
                WHERE index_symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                LIMIT 1
            """
            
            future_result = self.conn.execute(future_query, [index_symbol, future_timestamp]).fetchdf()
            
            if future_result.empty:
                logger.warning(f"⚠️ No future price data for {index_symbol} at {future_timestamp}")
                return {}
            
            future_close = future_result.iloc[0]['close_price']
            
            # Calculate label
            if future_close > current_close:
                label = 1  # UP
            else:
                label = -1  # DOWN
            
            # Calculate return percentage
            return_pct = (future_close - current_close) / current_close * 100
            
            label_data = {
                'timestamp': timestamp,
                'index_symbol': index_symbol,
                'label': label,
                'future_close': future_close,
                'current_close': current_close,
                'return_pct': return_pct
            }
            
            # Insert label into database
            self._insert_label(label_data)
            
            return label_data
            
        except Exception as e:
            logger.error(f"❌ Error creating labels: {e}")
            return {}
    
    def _insert_label(self, label_data: Dict[str, Any]):
        """Insert label data into database."""
        try:
            query = """
                INSERT INTO intraday_labels (timestamp, index_symbol, label, future_close, current_close, return_pct)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            self.conn.execute(query, [
                label_data['timestamp'],
                label_data['index_symbol'],
                label_data['label'],
                label_data['future_close'],
                label_data['current_close'],
                label_data['return_pct']
            ])
            
            logger.info(f"✅ Inserted label for {label_data['index_symbol']} at {label_data['timestamp']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert label: {e}")
    
    def create_complete_features(self, index_symbol: str, timestamp: datetime) -> Dict[str, float]:
        """
        Create complete feature set for a specific timestamp.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            timestamp: Timestamp for feature extraction
            
        Returns:
            Dictionary with all features
        """
        try:
            logger.info(f"🔧 Creating complete features for {index_symbol} at {timestamp}")
            
            features = {}
            
            # Add timestamp and index symbol
            features['timestamp'] = timestamp
            features['index_symbol'] = index_symbol
            
            # Get options features
            options_features = self.get_options_features(index_symbol, timestamp)
            features.update(options_features)
            
            # Get index features
            index_features = self.get_index_features(index_symbol, timestamp)
            features.update(index_features)
            
            # Get sentiment features
            sentiment_features = self.get_sentiment_features(timestamp)
            features.update(sentiment_features)
            
            logger.info(f"✅ Created {len(features)} features for {index_symbol}")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error creating complete features: {e}")
            return {}
    
    def get_training_data(self, index_symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get training data with features and labels for a date range.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            DataFrame with features and labels
        """
        try:
            logger.info(f"📊 Getting training data for {index_symbol} from {start_date} to {end_date}")
            
            # Get all timestamps in the date range
            query = """
                SELECT DISTINCT timestamp FROM intraday_index_data 
                WHERE index_symbol = ? 
                AND DATE(timestamp) BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            timestamps = self.conn.execute(query, [index_symbol, start_date, end_date]).fetchdf()
            
            if timestamps.empty:
                logger.warning(f"⚠️ No timestamps found for {index_symbol} in date range")
                return pd.DataFrame()
            
            # Create features for each timestamp
            all_features = []
            
            for _, row in timestamps.iterrows():
                timestamp = row['timestamp']
                
                # Create features
                features = self.create_complete_features(index_symbol, timestamp)
                
                if features:
                    # Create label
                    label_data = self.create_labels(index_symbol, timestamp)
                    
                    if label_data:
                        features['label'] = label_data['label']
                        features['return_pct'] = label_data['return_pct']
                        all_features.append(features)
            
            if all_features:
                training_data = pd.DataFrame(all_features)
                logger.info(f"✅ Created training data with {len(training_data)} samples")
                return training_data
            else:
                logger.warning("⚠️ No training data created")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error getting training data: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("🔒 Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
