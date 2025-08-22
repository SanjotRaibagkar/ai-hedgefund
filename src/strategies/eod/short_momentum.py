"""
Short Momentum Strategy for EOD Trading.
Implements bearish momentum-based strategies for swing trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger

from .momentum_indicators import MomentumIndicators
from .position_sizing import PositionSizing
from .risk_management import RiskManager


class ShortMomentumStrategy:
    """Short momentum strategy for bearish swing trading."""
    
    def __init__(self,
                 strategy_name: str = "Short Momentum",
                 min_signal_strength: float = 0.3,
                 min_momentum_score: float = 20.0,
                 min_volume_ratio: float = 1.5,
                 max_holding_period: int = 20,
                 **kwargs):
        """
        Initialize short momentum strategy.
        
        Args:
            strategy_name: Name of the strategy
            min_signal_strength: Minimum signal strength to enter position
            min_momentum_score: Minimum momentum score to enter position
            min_volume_ratio: Minimum volume ratio vs average
            max_holding_period: Maximum holding period in days
            **kwargs: Additional strategy parameters
        """
        self.strategy_name = strategy_name
        self.min_signal_strength = min_signal_strength
        self.min_momentum_score = min_momentum_score
        self.min_volume_ratio = min_volume_ratio
        self.max_holding_period = max_holding_period
        
        # Initialize components
        self.momentum_indicators = MomentumIndicators()
        self.position_sizing = PositionSizing(**kwargs.get('position_sizing', {}))
        self.risk_manager = RiskManager(**kwargs.get('risk_management', {}))
        
        # Strategy parameters
        self.parameters = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_signal_threshold': 0.001,
            'stochastic_oversold': 20,
            'stochastic_overbought': 80,
            'volume_ma_period': 20,
            'price_ma_short': 10,
            'price_ma_long': 50,
            'breakdown_threshold': 0.02,  # 2% breakdown
            'consolidation_threshold': 0.05,  # 5% consolidation
            **kwargs
        }
    
    def analyze_stock(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Analyze stock for short momentum opportunities.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if df.empty or len(df) < 50:
                return {
                    'ticker': ticker,
                    'signal': 'no_data',
                    'confidence': 0,
                    'reason': 'Insufficient data'
                }
            
            # Calculate momentum indicators
            df_with_indicators = self.momentum_indicators.calculate_all_indicators(df)
            
            # Get momentum signals
            momentum_signals = self.momentum_indicators.get_momentum_signals(df_with_indicators)
            overall_momentum = self.momentum_indicators.get_overall_momentum_score(momentum_signals)
            
            # Perform strategy-specific analysis
            analysis = self._perform_short_analysis(df_with_indicators, momentum_signals, overall_momentum)
            
            # Add ticker and momentum info
            analysis['ticker'] = ticker
            analysis['momentum_signals'] = momentum_signals
            analysis['overall_momentum'] = overall_momentum
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
            return {
                'ticker': ticker,
                'signal': 'error',
                'confidence': 0,
                'reason': f'Analysis failed: {e}'
            }
    
    def _perform_short_analysis(self, 
                              df: pd.DataFrame, 
                              momentum_signals: Dict[str, Dict[str, float]],
                              overall_momentum: Dict[str, float]) -> Dict[str, Any]:
        """Perform short-specific momentum analysis."""
        try:
            analysis = {
                'signal': 'hold',
                'confidence': 0,
                'reason': 'No clear signal',
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'position_size': None,
                'risk_reward_ratio': None
            }
            
            # Check basic momentum conditions
            if overall_momentum['direction'] != 'bearish':
                analysis['reason'] = f"Overall momentum is {overall_momentum['direction']}"
                return analysis
            
            if overall_momentum['score'] < self.min_momentum_score:
                analysis['reason'] = f"Momentum score {overall_momentum['score']:.1f} below threshold {self.min_momentum_score}"
                return analysis
            
            # Check volume conditions
            volume_analysis = self._analyze_volume(df)
            if not volume_analysis['volume_confirmed']:
                analysis['reason'] = f"Volume not confirmed: {volume_analysis['reason']}"
                return analysis
            
            # Check price action
            price_analysis = self._analyze_price_action(df)
            if not price_analysis['price_confirmed']:
                analysis['reason'] = f"Price action not confirmed: {price_analysis['reason']}"
                return analysis
            
            # Check trend conditions
            trend_analysis = self._analyze_trend(df)
            if not trend_analysis['trend_confirmed']:
                analysis['reason'] = f"Trend not confirmed: {trend_analysis['reason']}"
                return analysis
            
            # Calculate entry conditions
            entry_analysis = self._calculate_entry_conditions(df, momentum_signals)
            if not entry_analysis['entry_confirmed']:
                analysis['reason'] = f"Entry not confirmed: {entry_analysis['reason']}"
                return analysis            
            
            # All conditions met - generate sell signal
            current_price = df['close_price'].iloc[-1]
            
            # Calculate position sizing
            position_size = self.position_sizing.calculate_position_size(
                ticker=df.get('ticker', 'UNKNOWN').iloc[0] if 'ticker' in df.columns else 'UNKNOWN',
                current_price=current_price,
                signal_strength=overall_momentum['score'] / 100,
                volatility=df['close_price'].pct_change().std() if len(df) > 1 else 0.2,
                method='adaptive'
            )
            
            # Calculate stop loss
            stop_loss_info = self.risk_manager.calculate_stop_loss(
                ticker=df.get('ticker', 'UNKNOWN').iloc[0] if 'ticker' in df.columns else 'UNKNOWN',
                entry_price=current_price,
                position_type='short',
                method='adaptive',
                volatility=df['close_price'].pct_change().std() if len(df) > 1 else 0.2
            )
            
            # Calculate take profit
            take_profit_info = self.risk_manager.calculate_take_profit(
                ticker=df.get('ticker', 'UNKNOWN').iloc[0] if 'ticker' in df.columns else 'UNKNOWN',
                entry_price=current_price,
                stop_loss=stop_loss_info['stop_loss'],
                position_type='short',
                method='fixed_ratio',
                risk_reward_ratio=2.0
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                momentum_signals, overall_momentum, volume_analysis, 
                price_analysis, trend_analysis, entry_analysis
            )
            
            analysis.update({
                'signal': 'sell',
                'confidence': confidence,
                'reason': 'Short momentum signal confirmed',
                'entry_price': current_price,
                'stop_loss': stop_loss_info['stop_loss'],
                'take_profit': take_profit_info['take_profit'],
                'position_size': position_size,
                'risk_reward_ratio': take_profit_info['risk_reward_ratio'],
                'volume_analysis': volume_analysis,
                'price_analysis': price_analysis,
                'trend_analysis': trend_analysis,
                'entry_analysis': entry_analysis
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Short analysis failed: {e}")
            return {
                'signal': 'error',
                'confidence': 0,
                'reason': f'Analysis error: {e}'
            }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume conditions for short momentum."""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=self.parameters['volume_ma_period']).mean().iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Check for volume confirmation
            volume_confirmed = volume_ratio >= self.min_volume_ratio
            
            return {
                'volume_confirmed': volume_confirmed,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'reason': f"Volume ratio {volume_ratio:.2f} {'above' if volume_confirmed else 'below'} threshold {self.min_volume_ratio}"
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {
                'volume_confirmed': False,
                'reason': f'Volume analysis error: {e}'
            }
    
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action for short momentum."""
        try:
            current_price = df['close_price'].iloc[-1]
            prev_price = df['close_price'].iloc[-2] if len(df) > 1 else current_price
            
            # Check for price momentum (downward)
            price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            price_momentum = price_change < 0  # Negative for short
            
            # Check for lower highs and lower lows
            recent_highs = df['high_price'].tail(5)
            recent_lows = df['low_price'].tail(5)
            
            lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-2] if len(recent_highs) > 1 else True
            lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-2] if len(recent_lows) > 1 else True
            
            price_confirmed = price_momentum and lower_highs and lower_lows
            
            return {
                'price_confirmed': price_confirmed,
                'price_change': price_change,
                'price_momentum': price_momentum,
                'lower_highs': lower_highs,
                'lower_lows': lower_lows,
                'reason': f"Price momentum: {price_momentum}, Lower highs: {lower_highs}, Lower lows: {lower_lows}"
            }
            
        except Exception as e:
            logger.error(f"Price action analysis failed: {e}")
            return {
                'price_confirmed': False,
                'reason': f'Price action analysis error: {e}'
            }
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend conditions for short momentum."""
        try:
            # Calculate moving averages
            short_ma = df['close_price'].rolling(window=self.parameters['price_ma_short']).mean()
            long_ma = df['close_price'].rolling(window=self.parameters['price_ma_long']).mean()
            
            current_price = df['close_price'].iloc[-1]
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            
            # Check for downtrend
            price_below_short_ma = current_price < current_short_ma
            short_ma_below_long_ma = current_short_ma < current_long_ma
            long_ma_sloping_down = long_ma.iloc[-1] < long_ma.iloc[-5] if len(long_ma) > 5 else True
            
            trend_confirmed = price_below_short_ma and short_ma_below_long_ma and long_ma_sloping_down
            
            return {
                'trend_confirmed': trend_confirmed,
                'price_below_short_ma': price_below_short_ma,
                'short_ma_below_long_ma': short_ma_below_long_ma,
                'long_ma_sloping_down': long_ma_sloping_down,
                'short_ma': current_short_ma,
                'long_ma': current_long_ma,
                'reason': f"Trend: Price below short MA: {price_below_short_ma}, Short MA below long MA: {short_ma_below_long_ma}, Long MA sloping down: {long_ma_sloping_down}"
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {
                'trend_confirmed': False,
                'reason': f'Trend analysis error: {e}'
            }
    
    def _calculate_entry_conditions(self, df: pd.DataFrame, momentum_signals: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate entry conditions for short momentum."""
        try:
            entry_confirmed = True
            reasons = []
            
            # Check RSI conditions
            if 'rsi' in momentum_signals:
                rsi_signal = momentum_signals['rsi']
                if rsi_signal['signal'] == 'oversold':
                    entry_confirmed = False
                    reasons.append("RSI oversold")
                elif rsi_signal['signal'] == 'overbought':
                    reasons.append("RSI overbought - potential reversal")
            
            # Check MACD conditions
            if 'macd' in momentum_signals:
                macd_signal = momentum_signals['macd']
                if macd_signal['signal'] == 'bullish':
                    entry_confirmed = False
                    reasons.append("MACD bullish")
                elif macd_signal['histogram'] > self.parameters['macd_signal_threshold']:
                    entry_confirmed = False
                    reasons.append("MACD histogram positive")
            
            # Check Stochastic conditions
            if 'stochastic' in momentum_signals:
                stoch_signal = momentum_signals['stochastic']
                if stoch_signal['signal'] == 'oversold':
                    entry_confirmed = False
                    reasons.append("Stochastic oversold")
            
            # Check for breakdown or consolidation
            breakdown_analysis = self._check_breakdown_conditions(df)
            if breakdown_analysis['breakdown_detected']:
                reasons.append("Breakdown detected")
            elif breakdown_analysis['consolidation_detected']:
                reasons.append("Consolidation detected")
            
            return {
                'entry_confirmed': entry_confirmed,
                'reasons': reasons,
                'breakdown_analysis': breakdown_analysis,
                'reason': f"Entry {'confirmed' if entry_confirmed else 'not confirmed'}: {', '.join(reasons)}"
            }
            
        except Exception as e:
            logger.error(f"Entry conditions calculation failed: {e}")
            return {
                'entry_confirmed': False,
                'reason': f'Entry conditions error: {e}'
            }
    
    def _check_breakdown_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for breakdown or consolidation conditions."""
        try:
            current_price = df['close_price'].iloc[-1]
            
            # Calculate recent price range
            recent_high = df['high_price'].tail(10).max()
            recent_low = df['low_price'].tail(10).min()
            price_range = recent_high - recent_low
            range_percentage = price_range / recent_low if recent_low > 0 else 0
            
            # Check for breakdown
            breakdown_threshold = self.parameters['breakdown_threshold']
            consolidation_threshold = self.parameters['consolidation_threshold']
            
            breakdown_detected = current_price < recent_low * (1 + breakdown_threshold)
            consolidation_detected = range_percentage < consolidation_threshold
            
            return {
                'breakdown_detected': breakdown_detected,
                'consolidation_detected': consolidation_detected,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'price_range': price_range,
                'range_percentage': range_percentage
            }
            
        except Exception as e:
            logger.error(f"Breakdown conditions check failed: {e}")
            return {
                'breakdown_detected': False,
                'consolidation_detected': False
            }
    
    def _calculate_confidence_score(self, 
                                  momentum_signals: Dict[str, Dict[str, float]],
                                  overall_momentum: Dict[str, float],
                                  volume_analysis: Dict[str, Any],
                                  price_analysis: Dict[str, Any],
                                  trend_analysis: Dict[str, Any],
                                  entry_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the signal."""
        try:
            confidence = 0.0
            
            # Momentum score contribution (40%)
            momentum_confidence = min(1.0, overall_momentum['score'] / 100)
            confidence += momentum_confidence * 0.4
            
            # Volume confirmation contribution (20%)
            if volume_analysis.get('volume_confirmed', False):
                volume_confidence = min(1.0, volume_analysis.get('volume_ratio', 1) / 3)
                confidence += volume_confidence * 0.2
            
            # Price action contribution (20%)
            if price_analysis.get('price_confirmed', False):
                price_confidence = 1.0
                confidence += price_confidence * 0.2
            
            # Trend confirmation contribution (15%)
            if trend_analysis.get('trend_confirmed', False):
                trend_confidence = 1.0
                confidence += trend_confidence * 0.15
            
            # Entry conditions contribution (5%)
            if entry_analysis.get('entry_confirmed', False):
                entry_confidence = 1.0
                confidence += entry_confidence * 0.05
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Confidence score calculation failed: {e}")
            return 0.0
    
    def should_exit_position(self, 
                           position: Dict[str, Any],
                           current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if short position should be exited.
        
        Args:
            position: Current position information
            current_data: Current market data
            
        Returns:
            Dictionary with exit decision
        """
        try:
            if current_data.empty:
                return {'should_exit': False, 'reason': 'No data available'}
            
            current_price = current_data['close_price'].iloc[-1]
            entry_price = position.get('entry_price', current_price)
            stop_loss = position.get('stop_loss', entry_price * 1.05)
            take_profit = position.get('take_profit', entry_price * 0.90)
            
            # Check stop loss (price goes up for short position)
            if current_price >= stop_loss:
                return {
                    'should_exit': True,
                    'reason': 'stop_loss',
                    'exit_price': current_price,
                    'pnl': (entry_price - current_price) / entry_price
                }
            
            # Check take profit (price goes down for short position)
            if current_price <= take_profit:
                return {
                    'should_exit': True,
                    'reason': 'take_profit',
                    'exit_price': current_price,
                    'pnl': (entry_price - current_price) / entry_price
                }
            
            # Check momentum reversal
            momentum_analysis = self.analyze_stock(current_data, position.get('ticker', 'UNKNOWN'))
            if momentum_analysis['signal'] == 'buy':
                return {
                    'should_exit': True,
                    'reason': 'momentum_reversal',
                    'exit_price': current_price,
                    'pnl': (entry_price - current_price) / entry_price
                }
            
            # Check holding period
            entry_date = position.get('entry_date')
            if entry_date:
                days_held = (datetime.now() - entry_date).days
                if days_held >= self.max_holding_period:
                    return {
                        'should_exit': True,
                        'reason': 'max_holding_period',
                        'exit_price': current_price,
                        'pnl': (entry_price - current_price) / entry_price
                    }
            
            return {'should_exit': False, 'reason': 'Hold position'}
            
        except Exception as e:
            logger.error(f"Exit decision calculation failed: {e}")
            return {'should_exit': False, 'reason': f'Error: {e}'}
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary and parameters."""
        return {
            'strategy_name': self.strategy_name,
            'strategy_type': 'short_momentum',
            'parameters': self.parameters,
            'min_signal_strength': self.min_signal_strength,
            'min_momentum_score': self.min_momentum_score,
            'min_volume_ratio': self.min_volume_ratio,
            'max_holding_period': self.max_holding_period,
            'position_sizing_params': self.position_sizing.get_position_sizing_summary(),
            'risk_management_params': {
                'max_portfolio_risk': self.risk_manager.max_portfolio_risk,
                'max_position_risk': self.risk_manager.max_position_risk,
                'max_drawdown': self.risk_manager.max_drawdown,
                'max_positions': self.risk_manager.max_positions
            }
        } 