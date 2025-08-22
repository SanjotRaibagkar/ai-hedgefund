"""
Options Trading Strategies

This module contains strategies designed for options trading using options data,
Greeks, implied volatility, and options-specific indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class IVSkewStrategy(BaseStrategy):
    """Implied Volatility Skew strategy for options trading."""
    
    def __init__(self, skew_threshold: float = 0.1):
        """Initialize IV skew strategy."""
        super().__init__(
            name="IV Skew Strategy",
            description="Trades based on implied volatility skew patterns"
        )
        self.skew_threshold = skew_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for IV skew strategy."""
        required_fields = ['option_chain', 'current_price']
        if not all(field in data for field in required_fields):
            return False
        
        # Check if option chain has sufficient data
        option_chain = data.get('option_chain', pd.DataFrame())
        return not option_chain.empty and len(option_chain) > 5
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate IV skew signals."""
        option_chain = data['option_chain']
        current_price = data['current_price']
        
        if option_chain.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'No options data available',
                'metrics': {}
            }
        
        # Separate calls and puts
        calls = option_chain[option_chain['instrumentType'] == 'CE']
        puts = option_chain[option_chain['instrumentType'] == 'PE']
        
        if calls.empty or puts.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'Insufficient options data for skew analysis',
                'metrics': {}
            }
        
        # Calculate average IV for calls and puts
        avg_call_iv = calls['impliedVolatility'].mean()
        avg_put_iv = puts['impliedVolatility'].mean()
        
        # Calculate IV skew
        iv_skew = avg_put_iv - avg_call_iv
        
        # Generate signals based on skew
        if iv_skew > self.skew_threshold:
            signal = "BUY_PUTS"
            confidence = min(80, abs(iv_skew) * 100)
            reasoning = f"High put skew: {iv_skew:.3f} - market expects downside"
        elif iv_skew < -self.skew_threshold:
            signal = "BUY_CALLS"
            confidence = min(80, abs(iv_skew) * 100)
            reasoning = f"High call skew: {iv_skew:.3f} - market expects upside"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"Normal IV skew: {iv_skew:.3f}"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'iv_skew': iv_skew,
                'avg_call_iv': avg_call_iv,
                'avg_put_iv': avg_put_iv,
                'current_price': current_price
            }
        }


class GammaExposureStrategy(BaseStrategy):
    """Gamma exposure strategy for options trading."""
    
    def __init__(self, gamma_threshold: float = 0.01):
        """Initialize gamma exposure strategy."""
        super().__init__(
            name="Gamma Exposure Strategy",
            description="Trades based on gamma exposure and pin risk"
        )
        self.gamma_threshold = gamma_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for gamma exposure strategy."""
        required_fields = ['option_chain', 'current_price', 'expiry_date']
        if not all(field in data for field in required_fields):
            return False
        
        option_chain = data.get('option_chain', pd.DataFrame())
        return not option_chain.empty and 'gamma' in option_chain.columns
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gamma exposure signals."""
        option_chain = data['option_chain']
        current_price = data['current_price']
        expiry_date = data['expiry_date']
        
        if option_chain.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'No options data available',
                'metrics': {}
            }
        
        # Calculate days to expiry
        expiry_dt = pd.to_datetime(expiry_date)
        days_to_expiry = (expiry_dt - datetime.now()).days
        
        # Calculate total gamma exposure
        total_gamma = option_chain['gamma'].sum()
        
        # Find strikes with high gamma (near current price)
        near_strikes = option_chain[
            abs(option_chain['strikePrice'] - current_price) / current_price < 0.05
        ]
        
        if near_strikes.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'No options near current price',
                'metrics': {}
            }
        
        # Calculate gamma concentration
        gamma_concentration = near_strikes['gamma'].sum() / total_gamma if total_gamma > 0 else 0
        
        # Generate signals
        if days_to_expiry <= 1 and gamma_concentration > self.gamma_threshold:
            signal = "GAMMA_HEDGE"
            confidence = min(85, gamma_concentration * 100)
            reasoning = f"High gamma exposure near expiry: {gamma_concentration:.3f}"
        elif gamma_concentration > self.gamma_threshold * 2:
            signal = "GAMMA_SCALP"
            confidence = min(75, gamma_concentration * 80)
            reasoning = f"High gamma concentration: {gamma_concentration:.3f}"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"Normal gamma exposure: {gamma_concentration:.3f}"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'total_gamma': total_gamma,
                'gamma_concentration': gamma_concentration,
                'days_to_expiry': days_to_expiry,
                'current_price': current_price
            }
        }


class OptionsFlowStrategy(BaseStrategy):
    """Options flow analysis strategy."""
    
    def __init__(self, flow_threshold: float = 2.0):
        """Initialize options flow strategy."""
        super().__init__(
            name="Options Flow Strategy",
            description="Analyzes options flow for unusual activity"
        )
        self.flow_threshold = flow_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for options flow strategy."""
        required_fields = ['option_chain', 'volume_data']
        if not all(field in data for field in required_fields):
            return False
        
        option_chain = data.get('option_chain', pd.DataFrame())
        return not option_chain.empty and 'totalTradedVolume' in option_chain.columns
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate options flow signals."""
        option_chain = data['option_chain']
        volume_data = data.get('volume_data', {})
        
        if option_chain.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'No options data available',
                'metrics': {}
            }
        
        # Analyze unusual options activity
        calls = option_chain[option_chain['instrumentType'] == 'CE']
        puts = option_chain[option_chain['instrumentType'] == 'PE']
        
        # Calculate volume ratios
        avg_call_volume = calls['totalTradedVolume'].mean() if not calls.empty else 0
        avg_put_volume = puts['totalTradedVolume'].mean() if not puts.empty else 0
        
        # Find unusual volume
        unusual_calls = calls[calls['totalTradedVolume'] > avg_call_volume * self.flow_threshold]
        unusual_puts = puts[puts['totalTradedVolume'] > avg_put_volume * self.flow_threshold]
        
        # Generate signals
        if len(unusual_calls) > len(unusual_puts) and len(unusual_calls) > 0:
            signal = "BULLISH_FLOW"
            confidence = min(80, len(unusual_calls) * 10)
            reasoning = f"Unusual call activity: {len(unusual_calls)} high-volume calls"
        elif len(unusual_puts) > len(unusual_calls) and len(unusual_puts) > 0:
            signal = "BEARISH_FLOW"
            confidence = min(80, len(unusual_puts) * 10)
            reasoning = f"Unusual put activity: {len(unusual_puts)} high-volume puts"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = "Normal options flow"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'unusual_calls': len(unusual_calls),
                'unusual_puts': len(unusual_puts),
                'avg_call_volume': avg_call_volume,
                'avg_put_volume': avg_put_volume
            }
        }


class IronCondorStrategy(BaseStrategy):
    """Iron Condor options strategy."""
    
    def __init__(self, profit_target: float = 0.3, max_loss: float = 0.7):
        """Initialize Iron Condor strategy."""
        super().__init__(
            name="Iron Condor Strategy",
            description="Sells iron condors for premium collection"
        )
        self.profit_target = profit_target
        self.max_loss = max_loss
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for Iron Condor strategy."""
        required_fields = ['option_chain', 'current_price', 'volatility']
        return all(field in data for field in required_fields)
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Iron Condor signals."""
        option_chain = data['option_chain']
        current_price = data['current_price']
        volatility = data['volatility']
        
        if option_chain.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'No options data available',
                'metrics': {}
            }
        
        # Find suitable strikes for Iron Condor
        # Sell put spread below current price
        # Sell call spread above current price
        
        puts = option_chain[option_chain['instrumentType'] == 'PE']
        calls = option_chain[option_chain['instrumentType'] == 'CE']
        
        if puts.empty or calls.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'Insufficient options for Iron Condor',
                'metrics': {}
            }
        
        # Find strikes around current price
        put_strikes = puts[puts['strikePrice'] < current_price].sort_values('strikePrice', ascending=False)
        call_strikes = calls[calls['strikePrice'] > current_price].sort_values('strikePrice')
        
        if len(put_strikes) < 2 or len(call_strikes) < 2:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'Insufficient strikes for Iron Condor',
                'metrics': {}
            }
        
        # Calculate potential profit and loss
        put_credit = put_strikes.iloc[0]['lastPrice'] - put_strikes.iloc[1]['lastPrice']
        call_credit = call_strikes.iloc[0]['lastPrice'] - call_strikes.iloc[1]['lastPrice']
        total_credit = put_credit + call_credit
        
        # Calculate risk
        put_risk = put_strikes.iloc[0]['strikePrice'] - put_strikes.iloc[1]['strikePrice']
        call_risk = call_strikes.iloc[1]['strikePrice'] - call_strikes.iloc[0]['strikePrice']
        max_risk = max(put_risk, call_risk)
        
        # Calculate risk-reward ratio
        risk_reward_ratio = total_credit / max_risk if max_risk > 0 else 0
        
        # Generate signals
        if risk_reward_ratio > self.profit_target and volatility < 0.3:
            signal = "SELL_IRON_CONDOR"
            confidence = min(80, risk_reward_ratio * 50)
            reasoning = f"Good risk-reward: {risk_reward_ratio:.2f}, low volatility"
        elif risk_reward_ratio < self.max_loss:
            signal = "AVOID_IRON_CONDOR"
            confidence = 70
            reasoning = f"Poor risk-reward: {risk_reward_ratio:.2f}"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"Moderate risk-reward: {risk_reward_ratio:.2f}"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'risk_reward_ratio': risk_reward_ratio,
                'total_credit': total_credit,
                'max_risk': max_risk,
                'volatility': volatility,
                'put_credit': put_credit,
                'call_credit': call_credit
            }
        }


class StraddleStrategy(BaseStrategy):
    """Long Straddle options strategy."""
    
    def __init__(self, iv_percentile_threshold: float = 70):
        """Initialize Straddle strategy."""
        super().__init__(
            name="Long Straddle Strategy",
            description="Buys straddles when expecting large moves"
        )
        self.iv_percentile_threshold = iv_percentile_threshold
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate required data for Straddle strategy."""
        required_fields = ['option_chain', 'current_price', 'iv_percentile', 'earnings_date']
        return all(field in data for field in required_fields)
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Straddle signals."""
        option_chain = data['option_chain']
        current_price = data['current_price']
        iv_percentile = data['iv_percentile']
        earnings_date = data['earnings_date']
        
        if option_chain.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'No options data available',
                'metrics': {}
            }
        
        # Check if earnings are near
        earnings_dt = pd.to_datetime(earnings_date)
        days_to_earnings = (earnings_dt - datetime.now()).days
        
        # Find ATM options
        atm_calls = option_chain[
            (option_chain['instrumentType'] == 'CE') & 
            (abs(option_chain['strikePrice'] - current_price) / current_price < 0.02)
        ]
        
        atm_puts = option_chain[
            (option_chain['instrumentType'] == 'PE') & 
            (abs(option_chain['strikePrice'] - current_price) / current_price < 0.02)
        ]
        
        if atm_calls.empty or atm_puts.empty:
            return {
                'signal': 'HOLD',
                'confidence': 30,
                'reasoning': 'No ATM options available',
                'metrics': {}
            }
        
        # Calculate straddle cost
        call_cost = atm_calls.iloc[0]['lastPrice']
        put_cost = atm_puts.iloc[0]['lastPrice']
        straddle_cost = call_cost + put_cost
        
        # Generate signals
        if days_to_earnings <= 7 and iv_percentile < self.iv_percentile_threshold:
            signal = "BUY_STRADDLE"
            confidence = min(85, (7 - days_to_earnings) * 10 + (self.iv_percentile_threshold - iv_percentile))
            reasoning = f"Earnings in {days_to_earnings} days, low IV percentile: {iv_percentile}"
        elif iv_percentile < 30:
            signal = "BUY_STRADDLE"
            confidence = min(75, (30 - iv_percentile) * 2)
            reasoning = f"Very low IV percentile: {iv_percentile} - good time for straddle"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = f"IV percentile: {iv_percentile} - not optimal for straddle"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'straddle_cost': straddle_cost,
                'iv_percentile': iv_percentile,
                'days_to_earnings': days_to_earnings,
                'call_cost': call_cost,
                'put_cost': put_cost
            }
        } 