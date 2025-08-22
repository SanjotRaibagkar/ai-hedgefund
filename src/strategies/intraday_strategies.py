"""
Intraday Trading Strategies

This module contains strategies designed for intraday trading using real-time data,
market depth, and short-term technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum breakout strategy for intraday trading."""
    
    def __init__(self, breakout_threshold: float = 0.02, volume_threshold: float = 1.5):
        """Initialize momentum breakout strategy."""
        super().__init__(
            name="Momentum Breakout",
            description="Identifies momentum breakouts with volume confirmation"
        )
        self.breakout_threshold = breakout_threshold
        self.volume_threshold = volume_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for momentum breakout strategy."""
        required_fields = ['current_price', 'previous_close', 'volume', 'avg_volume']
        return all(field in data for field in required_fields)
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate momentum breakout signals."""
        current_price = data['current_price']
        previous_close = data['previous_close']
        volume = data['volume']
        avg_volume = data['avg_volume']
        
        # Calculate price change percentage
        price_change_pct = (current_price - previous_close) / previous_close
        
        # Calculate volume ratio
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
        
        # Generate signals
        if price_change_pct > self.breakout_threshold and volume_ratio > self.volume_threshold:
            signal = "BUY"
            confidence = min(90, abs(price_change_pct) * 100 + volume_ratio * 10)
            reasoning = f"Strong breakout: {price_change_pct:.2%} price increase with {volume_ratio:.1f}x volume"
        elif price_change_pct < -self.breakout_threshold and volume_ratio > self.volume_threshold:
            signal = "SELL"
            confidence = min(90, abs(price_change_pct) * 100 + volume_ratio * 10)
            reasoning = f"Strong breakdown: {price_change_pct:.2%} price decrease with {volume_ratio:.1f}x volume"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"No significant breakout: {price_change_pct:.2%} change, {volume_ratio:.1f}x volume"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'price_change_pct': price_change_pct,
                'volume_ratio': volume_ratio,
                'breakout_threshold': self.breakout_threshold,
                'volume_threshold': self.volume_threshold
            }
        }


class MarketDepthStrategy(BaseStrategy):
    """Market depth analysis strategy for intraday trading."""
    
    def __init__(self, bid_ask_ratio_threshold: float = 1.2):
        """Initialize market depth strategy."""
        super().__init__(
            name="Market Depth Analysis",
            description="Analyzes bid-ask imbalance and market depth for trading signals"
        )
        self.bid_ask_ratio_threshold = bid_ask_ratio_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for market depth strategy."""
        required_fields = ['market_depth', 'current_price']
        if not all(field in data for field in required_fields):
            return False
        
        # Check if market depth has required structure
        market_depth = data.get('market_depth', {})
        return 'bid' in market_depth and 'ask' in market_depth
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on market depth analysis."""
        market_depth = data['market_depth']
        current_price = data['current_price']
        
        # Analyze bid and ask orders
        bid_orders = market_depth.get('bid', [])
        ask_orders = market_depth.get('ask', [])
        
        if not bid_orders or not ask_orders:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'Insufficient market depth data',
                'metrics': {}
            }
        
        # Calculate total bid and ask volumes
        total_bid_volume = sum(order.get('quantity', 0) for order in bid_orders)
        total_ask_volume = sum(order.get('quantity', 0) for order in ask_orders)
        
        # Calculate bid-ask ratio
        bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1
        
        # Analyze price levels
        best_bid = max(order.get('price', 0) for order in bid_orders) if bid_orders else 0
        best_ask = min(order.get('price', float('inf')) for order in ask_orders) if ask_orders else float('inf')
        
        spread = best_ask - best_bid if best_ask != float('inf') else 0
        spread_pct = spread / current_price if current_price > 0 else 0
        
        # Generate signals
        if bid_ask_ratio > self.bid_ask_ratio_threshold:
            signal = "BUY"
            confidence = min(85, bid_ask_ratio * 30)
            reasoning = f"Strong buying pressure: {bid_ask_ratio:.2f} bid-ask ratio"
        elif bid_ask_ratio < (1 / self.bid_ask_ratio_threshold):
            signal = "SELL"
            confidence = min(85, (1 / bid_ask_ratio) * 30)
            reasoning = f"Strong selling pressure: {bid_ask_ratio:.2f} bid-ask ratio"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"Balanced order book: {bid_ask_ratio:.2f} bid-ask ratio"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'bid_ask_ratio': bid_ask_ratio,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'spread': spread,
                'spread_pct': spread_pct,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
        }


class VWAPStrategy(BaseStrategy):
    """VWAP-based intraday trading strategy."""
    
    def __init__(self, vwap_threshold: float = 0.005):
        """Initialize VWAP strategy."""
        super().__init__(
            name="VWAP Strategy",
            description="Trades based on price position relative to VWAP"
        )
        self.vwap_threshold = vwap_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for VWAP strategy."""
        required_fields = ['current_price', 'vwap']
        return all(field in data for field in required_fields)
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate VWAP-based signals."""
        current_price = data['current_price']
        vwap = data['vwap']
        
        if vwap <= 0:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'Invalid VWAP data',
                'metrics': {}
            }
        
        # Calculate price deviation from VWAP
        vwap_deviation = (current_price - vwap) / vwap
        
        # Generate signals
        if vwap_deviation > self.vwap_threshold:
            signal = "BUY"
            confidence = min(80, abs(vwap_deviation) * 100)
            reasoning = f"Price {vwap_deviation:.2%} above VWAP - bullish momentum"
        elif vwap_deviation < -self.vwap_threshold:
            signal = "SELL"
            confidence = min(80, abs(vwap_deviation) * 100)
            reasoning = f"Price {vwap_deviation:.2%} below VWAP - bearish momentum"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"Price near VWAP: {vwap_deviation:.2%} deviation"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'vwap_deviation': vwap_deviation,
                'current_price': current_price,
                'vwap': vwap,
                'vwap_threshold': self.vwap_threshold
            }
        }


class GapTradingStrategy(BaseStrategy):
    """Gap trading strategy for intraday trading."""
    
    def __init__(self, gap_threshold: float = 0.01):
        """Initialize gap trading strategy."""
        super().__init__(
            name="Gap Trading",
            description="Trades gaps at market open with fade or follow logic"
        )
        self.gap_threshold = gap_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for gap trading strategy."""
        required_fields = ['open_price', 'previous_close', 'current_price', 'market_open_time']
        return all(field in data for field in required_fields)
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gap trading signals."""
        open_price = data['open_price']
        previous_close = data['previous_close']
        current_price = data['current_price']
        market_open_time = data['market_open_time']
        
        # Calculate gap percentage
        gap_pct = (open_price - previous_close) / previous_close
        
        # Check if we're near market open (within first 30 minutes)
        current_time = datetime.now()
        time_since_open = current_time - market_open_time
        is_near_open = time_since_open.total_seconds() < 1800  # 30 minutes
        
        if not is_near_open:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'Not near market open - gap strategy not applicable',
                'metrics': {'gap_pct': gap_pct}
            }
        
        # Generate signals based on gap size and direction
        if abs(gap_pct) < self.gap_threshold:
            signal = "HOLD"
            confidence = 50
            reasoning = f"Small gap: {gap_pct:.2%} - no significant opportunity"
        elif gap_pct > self.gap_threshold:
            # Gap up - consider fade or follow
            if current_price < open_price:
                signal = "BUY"
                confidence = 70
                reasoning = f"Gap up {gap_pct:.2%} but price pulling back - fade opportunity"
            else:
                signal = "BUY"
                confidence = 60
                reasoning = f"Gap up {gap_pct:.2%} with momentum - follow opportunity"
        else:
            # Gap down - consider fade or follow
            if current_price > open_price:
                signal = "SELL"
                confidence = 70
                reasoning = f"Gap down {gap_pct:.2%} but price bouncing - fade opportunity"
            else:
                signal = "SELL"
                confidence = 60
                reasoning = f"Gap down {gap_pct:.2%} with momentum - follow opportunity"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'gap_pct': gap_pct,
                'open_price': open_price,
                'previous_close': previous_close,
                'current_price': current_price,
                'time_since_open_minutes': time_since_open.total_seconds() / 60
            }
        }


class IntradayMeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy for intraday trading."""
    
    def __init__(self, std_dev_threshold: float = 2.0, lookback_period: int = 20):
        """Initialize mean reversion strategy."""
        super().__init__(
            name="Intraday Mean Reversion",
            description="Trades mean reversion opportunities using Bollinger Bands"
        )
        self.std_dev_threshold = std_dev_threshold
        self.lookback_period = lookback_period
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for mean reversion strategy."""
        required_fields = ['price_history', 'current_price']
        if not all(field in data for field in required_fields):
            return False
        
        # Check if we have enough price history
        price_history = data.get('price_history', [])
        return len(price_history) >= self.lookback_period
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mean reversion signals."""
        price_history = data['price_history']
        current_price = data['current_price']
        
        if len(price_history) < self.lookback_period:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'Insufficient price history for mean reversion analysis',
                'metrics': {}
            }
        
        # Calculate Bollinger Bands
        prices = np.array(price_history[-self.lookback_period:])
        sma = np.mean(prices)
        std_dev = np.std(prices)
        
        upper_band = sma + (self.std_dev_threshold * std_dev)
        lower_band = sma - (self.std_dev_threshold * std_dev)
        
        # Calculate z-score
        z_score = (current_price - sma) / std_dev if std_dev > 0 else 0
        
        # Generate signals
        if z_score > self.std_dev_threshold:
            signal = "SELL"
            confidence = min(80, abs(z_score) * 20)
            reasoning = f"Price {z_score:.2f} std devs above mean - overbought"
        elif z_score < -self.std_dev_threshold:
            signal = "BUY"
            confidence = min(80, abs(z_score) * 20)
            reasoning = f"Price {z_score:.2f} std devs below mean - oversold"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"Price within normal range: {z_score:.2f} std devs from mean"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'z_score': z_score,
                'sma': sma,
                'std_dev': std_dev,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'current_price': current_price
            }
        } 