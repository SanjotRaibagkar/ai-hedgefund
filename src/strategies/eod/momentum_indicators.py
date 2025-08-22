"""
Momentum Indicators for EOD Strategies.
Calculates various momentum-based technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class MomentumIndicators:
    """Calculates momentum-based technical indicators for EOD strategies."""
    
    def __init__(self):
        """Initialize momentum indicators calculator."""
        self.indicators = {
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'stochastic': self._calculate_stochastic,
            'williams_r': self._calculate_williams_r,
            'cci': self._calculate_cci,
            'momentum': self._calculate_momentum,
            'roc': self._calculate_roc,
            'adx': self._calculate_adx,
            'mfi': self._calculate_mfi,
            'obv': self._calculate_obv,
            'vwap': self._calculate_vwap,
            'price_channels': self._calculate_price_channels,
            'bollinger_bands': self._calculate_bollinger_bands,
            'sma_20': lambda df: self._calculate_sma(df, 20),
            'sma_50': lambda df: self._calculate_sma(df, 50),
            'sma_200': lambda df: self._calculate_sma(df, 200),
            'atr_14': lambda df: self._calculate_atr(df, 14)
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        try:
            result_df = df.copy()
            
            # Ensure required columns exist
            required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            missing_cols = [col for col in required_cols if col not in result_df.columns]
            
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
                return result_df
            
            # Calculate each indicator
            for indicator_name, indicator_func in self.indicators.items():
                try:
                    indicator_data = indicator_func(result_df)
                    if isinstance(indicator_data, pd.DataFrame):
                        result_df = pd.concat([result_df, indicator_data], axis=1)
                    elif isinstance(indicator_data, pd.Series):
                        # Use the Series name if available, otherwise use indicator_name
                        column_name = indicator_data.name if indicator_data.name else indicator_name
                        result_df[column_name] = indicator_data
                except Exception as e:
                    logger.warning(f"Failed to calculate {indicator_name}: {e}")
                    continue
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = df['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return pd.Series(rsi, name=f'rsi_{period}')
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return pd.Series(index=df.index, name=f'rsi_{period}')
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            exp1 = df['close_price'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close_price'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            
            return pd.DataFrame({
                'macd': macd,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }, index=df.index)
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        try:
            lowest_low = df['low_price'].rolling(window=k_period).min()
            highest_high = df['high_price'].rolling(window=k_period).max()
            
            k_percent = 100 * ((df['close_price'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return pd.DataFrame({
                'stoch_k': k_percent,
                'stoch_d': d_percent
            })
        except Exception as e:
            logger.error(f"Stochastic calculation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        try:
            highest_high = df['high_price'].rolling(window=period).max()
            lowest_low = df['low_price'].rolling(window=period).min()
            williams_r = -100 * ((highest_high - df['close_price']) / (highest_high - lowest_low))
            return williams_r
        except Exception as e:
            logger.error(f"Williams %R calculation failed: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        try:
            typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci
        except Exception as e:
            logger.error(f"CCI calculation failed: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Momentum indicator."""
        try:
            momentum = df['close_price'] - df['close_price'].shift(period)
            return momentum
        except Exception as e:
            logger.error(f"Momentum calculation failed: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_roc(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Rate of Change."""
        try:
            roc = ((df['close_price'] - df['close_price'].shift(period)) / 
                   df['close_price'].shift(period)) * 100
            return roc
        except Exception as e:
            logger.error(f"ROC calculation failed: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        try:
            # Calculate True Range
            high_low = df['high_price'] - df['low_price']
            high_close = np.abs(df['high_price'] - df['close_price'].shift())
            low_close = np.abs(df['low_price'] - df['close_price'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Calculate Directional Movement
            up_move = df['high_price'] - df['high_price'].shift()
            down_move = df['low_price'].shift() - df['low_price']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth the values
            tr_smooth = true_range.rolling(window=period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth)
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = pd.Series(dx).rolling(window=period).mean()
            
            return pd.DataFrame({
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            })
        except Exception as e:
            logger.error(f"ADX calculation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        try:
            typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
            negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
            
            positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
            negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
            return mfi
        except Exception as e:
            logger.error(f"MFI calculation failed: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        try:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['volume'].iloc[0]
            
            for i in range(1, len(df)):
                if df['close_price'].iloc[i] > df['close_price'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close_price'].iloc[i] < df['close_price'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except Exception as e:
            logger.error(f"OBV calculation failed: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        try:
            typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
            vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
            return vwap
        except Exception as e:
            logger.error(f"VWAP calculation failed: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_price_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Price Channels (Highest High and Lowest Low)."""
        try:
            highest_high = df['high_price'].rolling(window=period).max()
            lowest_low = df['low_price'].rolling(window=period).min()
            
            return pd.DataFrame({
                'price_channel_high': highest_high,
                'price_channel_low': lowest_low
            })
        except Exception as e:
            logger.error(f"Price Channels calculation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            sma = df['close_price'].rolling(window=period).mean()
            std = df['close_price'].rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return pd.DataFrame({
                'bollinger_upper': upper_band,
                'bollinger_middle': sma,
                'bollinger_lower': lower_band,
                'bb_width': (upper_band - lower_band) / sma * 100
            }, index=df.index)
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _calculate_sma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        try:
            sma = df['close_price'].rolling(window=period).mean()
            return pd.Series(sma, name=f'sma_{period}')
        except Exception as e:
            logger.error(f"SMA calculation failed: {e}")
            return pd.Series(index=df.index, name=f'sma_{period}')
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high_low = df['high_price'] - df['low_price']
            high_close = np.abs(df['high_price'] - df['close_price'].shift())
            low_close = np.abs(df['low_price'] - df['close_price'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            return pd.Series(atr, name=f'atr_{period}')
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return pd.Series(index=df.index, name=f'atr_{period}')
    
    def get_momentum_signals(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get momentum signals for strategy decision making.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Dictionary of momentum signals
        """
        try:
            signals = {}
            
            # RSI signals
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                signals['rsi'] = {
                    'value': rsi,
                    'signal': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral',
                    'strength': abs(50 - rsi) / 50  # 0 to 1 scale
                }
            
            # MACD signals
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                macd_hist = df['macd_histogram'].iloc[-1]
                
                signals['macd'] = {
                    'value': macd,
                    'signal': 'bullish' if macd > macd_signal else 'bearish',
                    'strength': abs(macd_hist) / abs(macd) if macd != 0 else 0,
                    'histogram': macd_hist
                }
            
            # Stochastic signals
            if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                
                signals['stochastic'] = {
                    'value': stoch_k,
                    'signal': 'oversold' if stoch_k < 20 else 'overbought' if stoch_k > 80 else 'neutral',
                    'strength': abs(50 - stoch_k) / 50,
                    'k_d_diff': stoch_k - stoch_d
                }
            
            # Williams %R signals
            if 'williams_r' in df.columns:
                williams_r = df['williams_r'].iloc[-1]
                signals['williams_r'] = {
                    'value': williams_r,
                    'signal': 'oversold' if williams_r < -80 else 'overbought' if williams_r > -20 else 'neutral',
                    'strength': abs(-50 - williams_r) / 50
                }
            
            # CCI signals
            if 'cci' in df.columns:
                cci = df['cci'].iloc[-1]
                signals['cci'] = {
                    'value': cci,
                    'signal': 'oversold' if cci < -100 else 'overbought' if cci > 100 else 'neutral',
                    'strength': abs(cci) / 200 if abs(cci) <= 200 else 1
                }
            
            # Momentum signals
            if 'momentum' in df.columns:
                momentum = df['momentum'].iloc[-1]
                momentum_prev = df['momentum'].iloc[-2] if len(df) > 1 else 0
                
                signals['momentum'] = {
                    'value': momentum,
                    'signal': 'increasing' if momentum > momentum_prev else 'decreasing',
                    'strength': abs(momentum) / df['close_price'].iloc[-1] * 100
                }
            
            # ROC signals
            if 'roc' in df.columns:
                roc = df['roc'].iloc[-1]
                signals['roc'] = {
                    'value': roc,
                    'signal': 'bullish' if roc > 0 else 'bearish',
                    'strength': abs(roc) / 10  # Normalize to 0-1 scale
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get momentum signals: {e}")
            return {}
    
    def get_overall_momentum_score(self, signals: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate overall momentum score from all signals.
        
        Args:
            signals: Dictionary of momentum signals
            
        Returns:
            Overall momentum score and direction
        """
        try:
            if not signals:
                return {'score': 0, 'direction': 'neutral', 'strength': 0}
            
            bullish_signals = 0
            bearish_signals = 0
            total_strength = 0
            signal_count = 0
            
            # Weight different indicators
            weights = {
                'rsi': 0.15,
                'macd': 0.20,
                'stochastic': 0.15,
                'williams_r': 0.10,
                'cci': 0.10,
                'momentum': 0.15,
                'roc': 0.15
            }
            
            for indicator, signal_data in signals.items():
                if indicator in weights:
                    weight = weights[indicator]
                    strength = signal_data.get('strength', 0)
                    signal = signal_data.get('signal', 'neutral')
                    
                    if signal in ['bullish', 'oversold', 'increasing']:
                        bullish_signals += weight * strength
                    elif signal in ['bearish', 'overbought', 'decreasing']:
                        bearish_signals += weight * strength
                    
                    total_strength += weight * strength
                    signal_count += 1
            
            if signal_count == 0:
                return {'score': 0, 'direction': 'neutral', 'strength': 0}
            
            # Calculate overall score (-100 to +100)
            net_score = bullish_signals - bearish_signals
            overall_score = (net_score / total_strength) * 100 if total_strength > 0 else 0
            
            # Determine direction
            if overall_score > 20:
                direction = 'bullish'
            elif overall_score < -20:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            return {
                'score': overall_score,
                'direction': direction,
                'strength': total_strength / signal_count if signal_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate overall momentum score: {e}")
            return {'score': 0, 'direction': 'neutral', 'strength': 0} 