#!/usr/bin/env python3
"""
Intraday Stock Screener
Real-time intraday stock screening with breakout detection,
support/resistance levels, and short-term profit targets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.tools.enhanced_api import get_prices
from src.strategies.eod.momentum_indicators import MomentumIndicators


class IntradayStockScreener:
    """Intraday Stock Screener with real-time analysis."""
    
    def __init__(self, min_volume: int = 500000, min_price: float = 50.0):
        """
        Initialize Intraday Stock Screener.
        
        Args:
            min_volume: Minimum volume filter
            min_price: Minimum price filter
        """
        self.min_volume = min_volume
        self.min_price = min_price
        self.momentum_indicators = MomentumIndicators()
        self.logger = logging.getLogger(__name__)
        
    def screen_stocks(self, tickers: List[str], lookback_days: int = 10) -> Dict[str, Any]:
        """
        Screen stocks for intraday trading opportunities.
        
        Args:
            tickers: List of stock tickers to screen
            lookback_days: Number of days for historical analysis
            
        Returns:
            Dictionary with screening results
        """
        self.logger.info(f"Screening {len(tickers)} stocks for intraday opportunities")
        
        results = {
            'breakout_signals': [],
            'reversal_signals': [],
            'momentum_signals': [],
            'summary': {
                'total_stocks': len(tickers),
                'breakout_count': 0,
                'reversal_count': 0,
                'momentum_count': 0,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        for ticker in tickers:
            try:
                # Get recent data for intraday analysis
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                
                data = get_prices(ticker, start_date, end_date)
                if data is None or data.empty:
                    self.logger.warning(f"No data available for {ticker}")
                    continue
                
                # Analyze stock for intraday opportunities
                analysis = self._analyze_intraday_opportunities(data, ticker)
                if analysis:
                    signal_type = analysis['signal_type']
                    if signal_type == 'BREAKOUT':
                        results['breakout_signals'].append(analysis)
                        results['summary']['breakout_count'] += 1
                    elif signal_type == 'REVERSAL':
                        results['reversal_signals'].append(analysis)
                        results['summary']['reversal_count'] += 1
                    elif signal_type == 'MOMENTUM':
                        results['momentum_signals'].append(analysis)
                        results['summary']['momentum_count'] += 1
                        
            except Exception as e:
                self.logger.error(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort results by confidence
        for signal_type in ['breakout_signals', 'reversal_signals', 'momentum_signals']:
            results[signal_type] = sorted(
                results[signal_type], 
                key=lambda x: x['confidence'], 
                reverse=True
            )
        
        self.logger.info(f"Intraday screening complete: {results['summary']['breakout_count']} breakouts, {results['summary']['reversal_count']} reversals, {results['summary']['momentum_count']} momentum signals")
        return results
    
    def _analyze_intraday_opportunities(self, data: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Analyze individual stock for intraday opportunities.
        
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
            
            # Basic filters
            if current_volume < self.min_volume or current_price < self.min_price:
                return None
            
            # Check for different types of signals
            breakout_analysis = self._detect_breakouts(data, indicators)
            reversal_analysis = self._detect_reversals(data, indicators)
            momentum_analysis = self._detect_momentum(data, indicators)
            
            # Select the strongest signal
            signals = [
                (breakout_analysis, 'BREAKOUT'),
                (reversal_analysis, 'REVERSAL'),
                (momentum_analysis, 'MOMENTUM')
            ]
            
            best_signal = None
            best_score = 0
            
            for signal_data, signal_type in signals:
                if signal_data and signal_data['score'] > best_score:
                    best_signal = signal_data
                    best_score = signal_data['score']
                    best_type = signal_type
            
            if not best_signal or best_score < 3:  # Minimum score threshold
                return None
            
            # Calculate entry, stop loss, and targets
            entry, sl, targets = self._calculate_intraday_levels(data, current_price, best_type)
            
            # Calculate risk-reward ratio
            risk = abs(entry - sl)
            reward = abs(targets['T1'] - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Filter by minimum risk-reward ratio
            if rr_ratio < 1.5:  # Lower threshold for intraday
                return None
            
            return {
                'ticker': ticker,
                'signal_type': best_type,
                'confidence': min(100, best_score * 15),  # Scale to 0-100
                'current_price': current_price,
                'entry_price': entry,
                'stop_loss': sl,
                'targets': targets,
                'risk_reward_ratio': rr_ratio,
                'reasons': best_signal['reasons'],
                'volume': current_volume,
                'indicators': {
                    'rsi': indicators.get('rsi_14', [None])[-1],
                    'macd': indicators.get('macd', [None])[-1],
                    'sma_20': indicators.get('sma_20', [None])[-1],
                    'atr': indicators.get('atr_14', [None])[-1]
                },
                'support_resistance': self._get_support_resistance_levels(data)
            }
            
        except Exception as e:
            self.logger.error(f"Error in intraday analysis for {ticker}: {e}")
            return None
    
    def _detect_breakouts(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Optional[Dict[str, Any]]:
        """Detect price and volume breakouts."""
        score = 0
        reasons = []
        
        current_price = data['close_price'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Price breakout above resistance
        high_20 = data['high_price'].rolling(20).max().iloc[-2]  # Previous day's high
        if current_price > high_20 * 1.01:  # 1% above resistance
            score += 3
            reasons.append("Price breakout above resistance")
        
        # Volume breakout
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        if current_volume > avg_volume * 2.0:  # 2x average volume
            score += 2
            reasons.append("High volume breakout")
        
        # Gap up opening
        if len(data) >= 2:
            prev_close = data['close_price'].iloc[-2]
            gap_up = (current_price - prev_close) / prev_close
            if gap_up > 0.02:  # 2% gap up
                score += 2
                reasons.append("Gap up opening")
        
        # RSI momentum
        rsi = indicators.get('rsi_14', pd.Series([50]))
        if rsi.iloc[-1] > 60 and rsi.iloc[-1] > rsi.iloc[-2]:
            score += 1
            reasons.append("RSI momentum")
        
        return {'score': score, 'reasons': reasons} if score > 0 else None
    
    def _detect_reversals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Optional[Dict[str, Any]]:
        """Detect reversal patterns."""
        score = 0
        reasons = []
        
        current_price = data['close_price'].iloc[-1]
        
        # RSI reversal signals
        rsi = indicators.get('rsi_14', pd.Series([50]))
        if len(rsi) >= 3:
            # Oversold reversal
            if rsi.iloc[-3] < 30 and rsi.iloc[-2] > rsi.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-2]:
                score += 3
                reasons.append("RSI oversold reversal")
            
            # Overbought reversal
            elif rsi.iloc[-3] > 70 and rsi.iloc[-2] < rsi.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-2]:
                score += 3
                reasons.append("RSI overbought reversal")
        
        # MACD reversal
        macd = indicators.get('macd', pd.Series([0]))
        if len(macd) >= 3:
            # Bullish MACD crossover
            if macd.iloc[-3] < 0 and macd.iloc[-2] > macd.iloc[-3] and macd.iloc[-1] > 0:
                score += 2
                reasons.append("MACD bullish crossover")
            
            # Bearish MACD crossover
            elif macd.iloc[-3] > 0 and macd.iloc[-2] < macd.iloc[-3] and macd.iloc[-1] < 0:
                score += 2
                reasons.append("MACD bearish crossover")
        
        # Support/Resistance bounce
        if self._is_support_bounce(data):
            score += 2
            reasons.append("Support level bounce")
        elif self._is_resistance_rejection(data):
            score += 2
            reasons.append("Resistance level rejection")
        
        return {'score': score, 'reasons': reasons} if score > 0 else None
    
    def _detect_momentum(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Optional[Dict[str, Any]]:
        """Detect momentum continuation patterns."""
        score = 0
        reasons = []
        
        current_price = data['close_price'].iloc[-1]
        
        # Price momentum
        if len(data) >= 5:
            price_change_5 = (current_price - data['close_price'].iloc[-5]) / data['close_price'].iloc[-5]
            if abs(price_change_5) > 0.05:  # 5% move in 5 days
                score += 2
                reasons.append("Strong price momentum")
        
        # RSI momentum
        rsi = indicators.get('rsi_14', pd.Series([50]))
        if len(rsi) >= 3:
            if rsi.iloc[-1] > 60 and rsi.iloc[-1] > rsi.iloc[-2] > rsi.iloc[-3]:
                score += 2
                reasons.append("RSI uptrend momentum")
            elif rsi.iloc[-1] < 40 and rsi.iloc[-1] < rsi.iloc[-2] < rsi.iloc[-3]:
                score += 2
                reasons.append("RSI downtrend momentum")
        
        # Volume momentum
        volume = data['volume']
        if len(volume) >= 5:
            recent_avg = volume.iloc[-5:].mean()
            if volume.iloc[-1] > recent_avg * 1.5:
                score += 1
                reasons.append("Volume momentum")
        
        # Moving average alignment
        sma_20 = indicators.get('sma_20', pd.Series([0]))
        sma_50 = indicators.get('sma_50', pd.Series([0]))
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            score += 1
            reasons.append("Bullish MA alignment")
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            score += 1
            reasons.append("Bearish MA alignment")
        
        return {'score': score, 'reasons': reasons} if score > 0 else None
    
    def _calculate_intraday_levels(self, data: pd.DataFrame, current_price: float, signal_type: str) -> Tuple[float, float, Dict[str, float]]:
        """Calculate intraday entry, stop loss, and targets."""
        atr = self._calculate_atr(data)
        
        if signal_type == 'BREAKOUT':
            # Entry: Current price or slight pullback
            entry = current_price * 0.998  # 0.2% below current price
            
            # Stop loss: Below breakout level
            breakout_level = data['high_price'].rolling(20).max().iloc[-2]
            sl = breakout_level * 0.99  # 1% below breakout
            
            # Targets: Conservative for intraday
            targets = {
                'T1': entry + (atr * 1.0),  # 1 ATR
                'T2': entry + (atr * 1.5),  # 1.5 ATR
                'T3': entry + (atr * 2.0)   # 2 ATR
            }
            
        elif signal_type == 'REVERSAL':
            # Entry: Current price
            entry = current_price
            
            # Stop loss: Beyond reversal level
            if 'bullish' in str(data).lower():
                sl = entry * 0.985  # 1.5% below entry
            else:
                sl = entry * 1.015  # 1.5% above entry
            
            # Targets: Based on reversal strength
            targets = {
                'T1': entry + (atr * 0.8),  # 0.8 ATR
                'T2': entry + (atr * 1.2),  # 1.2 ATR
                'T3': entry + (atr * 1.8)   # 1.8 ATR
            }
            
        else:  # MOMENTUM
            # Entry: Current price
            entry = current_price
            
            # Stop loss: Recent swing low/high
            if current_price > data['close_price'].iloc[-2]:  # Uptrend
                sl = entry * 0.99  # 1% below entry
            else:  # Downtrend
                sl = entry * 1.01  # 1% above entry
            
            # Targets: Momentum-based
            targets = {
                'T1': entry + (atr * 0.6),  # 0.6 ATR
                'T2': entry + (atr * 1.0),  # 1.0 ATR
                'T3': entry + (atr * 1.5)   # 1.5 ATR
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
        
        return atr if not pd.isna(atr) else data['close_price'].iloc[-1] * 0.015
    
    def _get_support_resistance_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get key support and resistance levels."""
        high = data['high_price']
        low = data['low_price']
        
        resistance = high.rolling(20).max().iloc[-1]
        support = low.rolling(20).min().iloc[-1]
        
        return {
            'resistance': resistance,
            'support': support,
            'pivot': (resistance + support) / 2
        }
    
    def _is_support_bounce(self, data: pd.DataFrame) -> bool:
        """Check if price is bouncing from support level."""
        current_price = data['close_price'].iloc[-1]
        support = data['low_price'].rolling(20).min().iloc[-1]
        
        return abs(current_price - support) / support < 0.01  # Within 1%
    
    def _is_resistance_rejection(self, data: pd.DataFrame) -> bool:
        """Check if price is being rejected from resistance level."""
        current_price = data['close_price'].iloc[-1]
        resistance = data['high_price'].rolling(20).max().iloc[-1]
        
        return abs(current_price - resistance) / resistance < 0.01  # Within 1%
    
    def get_screener_summary(self) -> Dict[str, Any]:
        """Get screener configuration and capabilities."""
        return {
            'name': 'Intraday Stock Screener',
            'description': 'Real-time intraday stock screening with breakout and reversal detection',
            'features': [
                'Breakout detection (price and volume)',
                'Reversal pattern recognition',
                'Momentum continuation analysis',
                'Support/Resistance level identification',
                'Intraday entry, stop loss, and target calculation'
            ],
            'filters': {
                'min_volume': self.min_volume,
                'min_price': self.min_price
            },
            'signal_types': [
                'BREAKOUT - Price/volume breakouts',
                'REVERSAL - Pattern reversals',
                'MOMENTUM - Trend continuation'
            ],
            'indicators_used': [
                'RSI (14)',
                'MACD',
                'SMA (20, 50)',
                'ATR (14)',
                'Volume Analysis',
                'Support/Resistance Levels'
            ]
        } 