#!/usr/bin/env python3
"""
EOD Stock Screener
Comprehensive end-of-day stock screening with bullish/bearish signals,
entry points, stop loss, and target calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.tools.enhanced_api import get_prices
from src.strategies.eod.momentum_indicators import MomentumIndicators


class EODStockScreener:
    """End-of-Day Stock Screener with comprehensive analysis."""
    
    def __init__(self, risk_reward_ratio: float = 2.0, min_volume: int = 1000000):
        """
        Initialize EOD Stock Screener.
        
        Args:
            risk_reward_ratio: Minimum risk-reward ratio for signals
            min_volume: Minimum volume filter
        """
        self.risk_reward_ratio = risk_reward_ratio
        self.min_volume = min_volume
        self.momentum_indicators = MomentumIndicators()
        self.logger = logging.getLogger(__name__)
        
    def screen_stocks(self, tickers: List[str], lookback_days: int = 30) -> Dict[str, Any]:
        """
        Screen stocks for EOD trading opportunities.
        
        Args:
            tickers: List of stock tickers to screen
            lookback_days: Number of days for historical analysis
            
        Returns:
            Dictionary with screening results
        """
        self.logger.info(f"Screening {len(tickers)} stocks for EOD opportunities")
        
        results = {
            'bullish_signals': [],
            'bearish_signals': [],
            'summary': {
                'total_stocks': len(tickers),
                'bullish_count': 0,
                'bearish_count': 0,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        for ticker in tickers:
            try:
                # Get historical data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                
                data = get_prices(ticker, start_date, end_date)
                if data is None or data.empty:
                    self.logger.warning(f"No data available for {ticker}")
                    continue
                
                # Analyze stock
                analysis = self._analyze_stock(data, ticker)
                if analysis:
                    if analysis['signal'] == 'BULLISH':
                        results['bullish_signals'].append(analysis)
                        results['summary']['bullish_count'] += 1
                    elif analysis['signal'] == 'BEARISH':
                        results['bearish_signals'].append(analysis)
                        results['summary']['bearish_count'] += 1
                        
            except Exception as e:
                self.logger.error(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort results by confidence
        results['bullish_signals'] = sorted(
            results['bullish_signals'], 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        results['bearish_signals'] = sorted(
            results['bearish_signals'], 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        self.logger.info(f"Screening complete: {results['summary']['bullish_count']} bullish, {results['summary']['bearish_count']} bearish signals")
        return results
    
    def _analyze_stock(self, data: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Analyze individual stock for EOD signals.
        
        Args:
            data: Historical price data
            ticker: Stock ticker
            
        Returns:
            Analysis result dictionary or None
        """
        try:
            # Calculate technical indicators
            indicators = self.momentum_indicators.calculate_all_indicators(data)
            
            # Get current price and volume
            current_price = data['close_price'].iloc[-1]
            current_volume = data['volume'].iloc[-1]
            
            # Volume filter
            if current_volume < self.min_volume:
                return None
            
            # Generate signals
            bullish_signals = self._get_bullish_signals(indicators, data)
            bearish_signals = self._get_bearish_signals(indicators, data)
            
            # Determine primary signal
            if bullish_signals['score'] > bearish_signals['score']:
                signal = 'BULLISH'
                signal_score = bullish_signals['score']
                signal_reasons = bullish_signals['reasons']
                entry, sl, targets = self._calculate_bullish_levels(data, current_price)
            elif bearish_signals['score'] > bullish_signals['score']:
                signal = 'BEARISH'
                signal_score = bearish_signals['score']
                signal_reasons = bearish_signals['reasons']
                entry, sl, targets = self._calculate_bearish_levels(data, current_price)
            else:
                return None  # No clear signal
            
            # Calculate risk-reward ratio
            risk = abs(entry - sl)
            reward = abs(targets['T1'] - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Filter by risk-reward ratio
            if rr_ratio < self.risk_reward_ratio:
                return None
            
            # Calculate confidence
            confidence = min(100, signal_score * 20)  # Scale to 0-100
            
            return {
                'ticker': ticker,
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'entry_price': entry,
                'stop_loss': sl,
                'targets': targets,
                'risk_reward_ratio': rr_ratio,
                'reasons': signal_reasons,
                'volume': current_volume,
                'indicators': {
                    'rsi': indicators.get('rsi_14', [None])[-1],
                    'macd': indicators.get('macd', [None])[-1],
                    'sma_20': indicators.get('sma_20', [None])[-1],
                    'sma_50': indicators.get('sma_50', [None])[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in stock analysis for {ticker}: {e}")
            return None
    
    def _get_bullish_signals(self, indicators: Dict[str, pd.Series], data: pd.DataFrame) -> Dict[str, Any]:
        """Get bullish signal score and reasons."""
        score = 0
        reasons = []
        
        # RSI oversold bounce
        rsi = indicators.get('rsi_14', pd.Series([50]))
        if rsi.iloc[-1] < 30 and rsi.iloc[-1] > rsi.iloc[-2]:
            score += 2
            reasons.append("RSI oversold bounce")
        
        # MACD bullish crossover
        macd = indicators.get('macd', pd.Series([0]))
        if len(macd) >= 2 and macd.iloc[-1] > macd.iloc[-2] and macd.iloc[-2] < 0:
            score += 2
            reasons.append("MACD bullish crossover")
        
        # Price above moving averages
        sma_20 = indicators.get('sma_20', pd.Series([0]))
        sma_50 = indicators.get('sma_50', pd.Series([0]))
        current_price = data['close_price'].iloc[-1]
        
        if current_price > sma_20.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
            score += 1
            reasons.append("Price above moving averages")
        
        # Volume breakout
        volume = data['volume']
        avg_volume = volume.rolling(20).mean()
        if volume.iloc[-1] > avg_volume.iloc[-1] * 1.5:
            score += 1
            reasons.append("Volume breakout")
        
        # Support bounce
        if self._is_support_bounce(data):
            score += 2
            reasons.append("Support level bounce")
        
        return {'score': score, 'reasons': reasons}
    
    def _get_bearish_signals(self, indicators: Dict[str, pd.Series], data: pd.DataFrame) -> Dict[str, Any]:
        """Get bearish signal score and reasons."""
        score = 0
        reasons = []
        
        # RSI overbought reversal
        rsi = indicators.get('rsi_14', pd.Series([50]))
        if rsi.iloc[-1] > 70 and rsi.iloc[-1] < rsi.iloc[-2]:
            score += 2
            reasons.append("RSI overbought reversal")
        
        # MACD bearish crossover
        macd = indicators.get('macd', pd.Series([0]))
        if len(macd) >= 2 and macd.iloc[-1] < macd.iloc[-2] and macd.iloc[-2] > 0:
            score += 2
            reasons.append("MACD bearish crossover")
        
        # Price below moving averages
        sma_20 = indicators.get('sma_20', pd.Series([0]))
        sma_50 = indicators.get('sma_50', pd.Series([0]))
        current_price = data['close_price'].iloc[-1]
        
        if current_price < sma_20.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
            score += 1
            reasons.append("Price below moving averages")
        
        # Volume breakdown
        volume = data['volume']
        avg_volume = volume.rolling(20).mean()
        if volume.iloc[-1] > avg_volume.iloc[-1] * 1.5:
            score += 1
            reasons.append("High volume breakdown")
        
        # Resistance rejection
        if self._is_resistance_rejection(data):
            score += 2
            reasons.append("Resistance level rejection")
        
        return {'score': score, 'reasons': reasons}
    
    def _calculate_bullish_levels(self, data: pd.DataFrame, current_price: float) -> Tuple[float, float, Dict[str, float]]:
        """Calculate bullish entry, stop loss, and targets."""
        # Entry: Current price or slight pullback
        entry = current_price * 0.995  # 0.5% below current price
        
        # Stop loss: Recent support level
        low_20 = data['low_price'].rolling(20).min().iloc[-1]
        sl = low_20 * 0.98  # 2% below support
        
        # Targets based on ATR
        atr = self._calculate_atr(data)
        targets = {
            'T1': entry + (atr * 1.5),  # 1.5 ATR
            'T2': entry + (atr * 2.5),  # 2.5 ATR
            'T3': entry + (atr * 4.0)   # 4.0 ATR
        }
        
        return entry, sl, targets
    
    def _calculate_bearish_levels(self, data: pd.DataFrame, current_price: float) -> Tuple[float, float, Dict[str, float]]:
        """Calculate bearish entry, stop loss, and targets."""
        # Entry: Current price or slight bounce
        entry = current_price * 1.005  # 0.5% above current price
        
        # Stop loss: Recent resistance level
        high_20 = data['high_price'].rolling(20).max().iloc[-1]
        sl = high_20 * 1.02  # 2% above resistance
        
        # Targets based on ATR
        atr = self._calculate_atr(data)
        targets = {
            'T1': entry - (atr * 1.5),  # 1.5 ATR
            'T2': entry - (atr * 2.5),  # 2.5 ATR
            'T3': entry - (atr * 4.0)   # 4.0 ATR
        }
        
        return entry, sl, targets
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['high_price']
        low = data['low_price']
        close = data['close_price']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else current_price * 0.02
    
    def _is_support_bounce(self, data: pd.DataFrame) -> bool:
        """Check if price is bouncing from support level."""
        current_price = data['close_price'].iloc[-1]
        low_20 = data['low_price'].rolling(20).min().iloc[-1]
        
        # Check if current price is near recent low
        return abs(current_price - low_20) / low_20 < 0.02  # Within 2%
    
    def _is_resistance_rejection(self, data: pd.DataFrame) -> bool:
        """Check if price is being rejected from resistance level."""
        current_price = data['close_price'].iloc[-1]
        high_20 = data['high_price'].rolling(20).max().iloc[-1]
        
        # Check if current price is near recent high
        return abs(current_price - high_20) / high_20 < 0.02  # Within 2%
    
    def get_screener_summary(self) -> Dict[str, Any]:
        """Get screener configuration and capabilities."""
        return {
            'name': 'EOD Stock Screener',
            'description': 'Comprehensive end-of-day stock screening with technical analysis',
            'features': [
                'Bullish/Bearish signal detection',
                'Entry, Stop Loss, and Target calculation',
                'Risk-Reward ratio filtering',
                'Volume-based filtering',
                'Technical indicator analysis'
            ],
            'filters': {
                'min_risk_reward_ratio': self.risk_reward_ratio,
                'min_volume': self.min_volume
            },
            'indicators_used': [
                'RSI (14)',
                'MACD',
                'SMA (20, 50)',
                'ATR (14)',
                'Volume Analysis'
            ]
        } 