"""
Position Sizing for EOD Momentum Strategies.
Handles position sizing calculations and risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class PositionSizing:
    """Handles position sizing calculations for EOD momentum strategies."""
    
    def __init__(self, 
                 account_size: float = 100000,
                 max_position_size: float = 0.1,  # 10% of account
                 max_risk_per_trade: float = 0.02,  # 2% risk per trade
                 volatility_lookback: int = 20):
        """
        Initialize position sizing calculator.
        
        Args:
            account_size: Total account size in currency
            max_position_size: Maximum position size as fraction of account
            max_risk_per_trade: Maximum risk per trade as fraction of account
            volatility_lookback: Period for volatility calculation
        """
        self.account_size = account_size
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.volatility_lookback = volatility_lookback
        
        # Position sizing methods
        self.sizing_methods = {
            'fixed': self._fixed_position_sizing,
            'kelly': self._kelly_criterion_sizing,
            'volatility': self._volatility_based_sizing,
            'risk_parity': self._risk_parity_sizing,
            'momentum': self._momentum_based_sizing,
            'adaptive': self._adaptive_position_sizing
        }
    
    def calculate_position_size(self, 
                              ticker: str,
                              current_price: float,
                              signal_strength: float,
                              volatility: float = None,
                              method: str = 'adaptive',
                              **kwargs) -> Dict[str, float]:
        """
        Calculate position size for a trade.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            signal_strength: Signal strength (-1 to 1)
            volatility: Stock volatility (optional)
            method: Position sizing method
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with position sizing details
        """
        try:
            if method not in self.sizing_methods:
                logger.warning(f"Unknown position sizing method: {method}. Using adaptive.")
                method = 'adaptive'
            
            sizing_func = self.sizing_methods[method]
            position_size = sizing_func(
                ticker=ticker,
                current_price=current_price,
                signal_strength=signal_strength,
                volatility=volatility,
                **kwargs
            )
            
            # Apply account-level constraints
            position_size = self._apply_constraints(position_size, current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return self._get_default_position_size(current_price)
    
    def _fixed_position_sizing(self, 
                              ticker: str,
                              current_price: float,
                              signal_strength: float,
                              **kwargs) -> Dict[str, float]:
        """Fixed position sizing based on account percentage."""
        try:
            # Use signal strength to determine position size
            base_size = self.account_size * self.max_position_size
            adjusted_size = base_size * abs(signal_strength)
            
            shares = int(adjusted_size / current_price)
            position_value = shares * current_price
            
            return {
                'shares': shares,
                'position_value': position_value,
                'account_percentage': position_value / self.account_size,
                'method': 'fixed',
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"Fixed position sizing failed: {e}")
            return self._get_default_position_size(current_price)
    
    def _kelly_criterion_sizing(self, 
                               ticker: str,
                               current_price: float,
                               signal_strength: float,
                               win_rate: float = 0.6,
                               avg_win: float = 0.1,
                               avg_loss: float = 0.05,
                               **kwargs) -> Dict[str, float]:
        """Kelly Criterion position sizing."""
        try:
            # Kelly Criterion: f = (bp - q) / b
            # where: b = odds received, p = probability of win, q = probability of loss
            b = avg_win / avg_loss  # odds received
            p = win_rate  # probability of win
            q = 1 - win_rate  # probability of loss
            
            kelly_fraction = (b * p - q) / b
            
            # Apply signal strength and constraints
            kelly_fraction = kelly_fraction * abs(signal_strength)
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
            
            position_value = self.account_size * kelly_fraction
            shares = int(position_value / current_price)
            
            return {
                'shares': shares,
                'position_value': shares * current_price,
                'account_percentage': (shares * current_price) / self.account_size,
                'method': 'kelly',
                'kelly_fraction': kelly_fraction,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"Kelly criterion sizing failed: {e}")
            return self._get_default_position_size(current_price)
    
    def _volatility_based_sizing(self, 
                                ticker: str,
                                current_price: float,
                                signal_strength: float,
                                volatility: float = None,
                                **kwargs) -> Dict[str, float]:
        """Volatility-based position sizing."""
        try:
            if volatility is None:
                volatility = 0.2  # Default 20% volatility
            
            # Inverse relationship: higher volatility = smaller position
            volatility_factor = 1 / (1 + volatility)
            
            # Base position size
            base_size = self.account_size * self.max_position_size
            adjusted_size = base_size * volatility_factor * abs(signal_strength)
            
            shares = int(adjusted_size / current_price)
            position_value = shares * current_price
            
            return {
                'shares': shares,
                'position_value': position_value,
                'account_percentage': position_value / self.account_size,
                'method': 'volatility',
                'volatility_factor': volatility_factor,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"Volatility-based sizing failed: {e}")
            return self._get_default_position_size(current_price)
    
    def _risk_parity_sizing(self, 
                           ticker: str,
                           current_price: float,
                           signal_strength: float,
                           volatility: float = None,
                           **kwargs) -> Dict[str, float]:
        """Risk parity position sizing."""
        try:
            if volatility is None:
                volatility = 0.2  # Default 20% volatility
            
            # Risk parity: equal risk contribution
            # Position size inversely proportional to volatility
            risk_budget = self.account_size * self.max_risk_per_trade
            position_risk = risk_budget / volatility
            
            # Convert risk to position size
            position_value = position_risk * abs(signal_strength)
            shares = int(position_value / current_price)
            
            return {
                'shares': shares,
                'position_value': shares * current_price,
                'account_percentage': (shares * current_price) / self.account_size,
                'method': 'risk_parity',
                'position_risk': position_risk,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"Risk parity sizing failed: {e}")
            return self._get_default_position_size(current_price)
    
    def _momentum_based_sizing(self, 
                              ticker: str,
                              current_price: float,
                              signal_strength: float,
                              momentum_score: float = None,
                              **kwargs) -> Dict[str, float]:
        """Momentum-based position sizing."""
        try:
            if momentum_score is None:
                momentum_score = abs(signal_strength)
            
            # Normalize momentum score to 0-1 range
            momentum_factor = min(1.0, abs(momentum_score) / 100)
            
            # Base position size
            base_size = self.account_size * self.max_position_size
            adjusted_size = base_size * momentum_factor
            
            shares = int(adjusted_size / current_price)
            position_value = shares * current_price
            
            return {
                'shares': shares,
                'position_value': position_value,
                'account_percentage': position_value / self.account_size,
                'method': 'momentum',
                'momentum_factor': momentum_factor,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"Momentum-based sizing failed: {e}")
            return self._get_default_position_size(current_price)
    
    def _adaptive_position_sizing(self, 
                                 ticker: str,
                                 current_price: float,
                                 signal_strength: float,
                                 volatility: float = None,
                                 momentum_score: float = None,
                                 market_conditions: str = 'normal',
                                 **kwargs) -> Dict[str, float]:
        """Adaptive position sizing combining multiple factors."""
        try:
            # Base position size
            base_size = self.account_size * self.max_position_size
            
            # Signal strength factor
            signal_factor = abs(signal_strength)
            
            # Volatility factor (inverse relationship)
            if volatility is None:
                volatility = 0.2
            volatility_factor = 1 / (1 + volatility)
            
            # Momentum factor
            if momentum_score is None:
                momentum_factor = signal_factor
            else:
                momentum_factor = min(1.0, abs(momentum_score) / 100)
            
            # Market conditions factor
            market_factors = {
                'bull': 1.2,
                'normal': 1.0,
                'bear': 0.8,
                'volatile': 0.6
            }
            market_factor = market_factors.get(market_conditions, 1.0)
            
            # Combine all factors
            combined_factor = signal_factor * volatility_factor * momentum_factor * market_factor
            
            # Calculate final position size
            adjusted_size = base_size * combined_factor
            shares = int(adjusted_size / current_price)
            position_value = shares * current_price
            
            return {
                'shares': shares,
                'position_value': position_value,
                'account_percentage': position_value / self.account_size,
                'method': 'adaptive',
                'signal_factor': signal_factor,
                'volatility_factor': volatility_factor,
                'momentum_factor': momentum_factor,
                'market_factor': market_factor,
                'combined_factor': combined_factor,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"Adaptive position sizing failed: {e}")
            return self._get_default_position_size(current_price)
    
    def _apply_constraints(self, position_size: Dict[str, float], current_price: float) -> Dict[str, float]:
        """Apply account-level constraints to position size."""
        try:
            # Ensure position doesn't exceed maximum account percentage
            max_position_value = self.account_size * self.max_position_size
            if position_size['position_value'] > max_position_value:
                position_size['position_value'] = max_position_value
                position_size['shares'] = int(max_position_value / current_price)
                position_size['account_percentage'] = max_position_value / self.account_size
            
            # Ensure minimum position size
            min_position_value = self.account_size * 0.01  # 1% minimum
            if position_size['position_value'] < min_position_value:
                position_size['position_value'] = min_position_value
                position_size['shares'] = int(min_position_value / current_price)
                position_size['account_percentage'] = min_position_value / self.account_size
            
            # Ensure shares is at least 1
            if position_size['shares'] < 1:
                position_size['shares'] = 1
                position_size['position_value'] = current_price
                position_size['account_percentage'] = current_price / self.account_size
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to apply constraints: {e}")
            return position_size
    
    def _get_default_position_size(self, current_price: float) -> Dict[str, float]:
        """Get default position size when calculation fails."""
        try:
            default_value = self.account_size * 0.05  # 5% default
            shares = int(default_value / current_price)
            
            return {
                'shares': max(1, shares),
                'position_value': max(1, shares) * current_price,
                'account_percentage': (max(1, shares) * current_price) / self.account_size,
                'method': 'default',
                'signal_strength': 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get default position size: {e}")
            return {
                'shares': 1,
                'position_value': current_price,
                'account_percentage': current_price / self.account_size,
                'method': 'fallback',
                'signal_strength': 0
            }
    
    def calculate_portfolio_allocation(self, 
                                     positions: List[Dict[str, any]],
                                     target_allocation: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate portfolio allocation across multiple positions.
        
        Args:
            positions: List of position dictionaries
            target_allocation: Target allocation percentages
            
        Returns:
            Dictionary with allocation details
        """
        try:
            total_portfolio_value = sum(pos.get('position_value', 0) for pos in positions)
            
            if total_portfolio_value == 0:
                return {}
            
            allocation = {}
            
            for position in positions:
                ticker = position.get('ticker', 'UNKNOWN')
                position_value = position.get('position_value', 0)
                
                allocation[ticker] = {
                    'position_value': position_value,
                    'portfolio_percentage': position_value / total_portfolio_value,
                    'account_percentage': position_value / self.account_size,
                    'shares': position.get('shares', 0)
                }
            
            # Check against target allocation if provided
            if target_allocation:
                for ticker, target_pct in target_allocation.items():
                    if ticker in allocation:
                        current_pct = allocation[ticker]['portfolio_percentage']
                        allocation[ticker]['target_deviation'] = current_pct - target_pct
            
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio allocation: {e}")
            return {}
    
    def calculate_risk_metrics(self, 
                              positions: List[Dict[str, any]],
                              volatilities: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            positions: List of position dictionaries
            volatilities: Dictionary of stock volatilities
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            total_portfolio_value = sum(pos.get('position_value', 0) for pos in positions)
            
            if total_portfolio_value == 0:
                return {
                    'total_risk': 0,
                    'portfolio_volatility': 0,
                    'max_drawdown_risk': 0,
                    'concentration_risk': 0
                }
            
            # Calculate portfolio volatility
            portfolio_volatility = 0
            if volatilities:
                for position in positions:
                    ticker = position.get('ticker', 'UNKNOWN')
                    position_value = position.get('position_value', 0)
                    weight = position_value / total_portfolio_value
                    volatility = volatilities.get(ticker, 0.2)
                    portfolio_volatility += (weight * volatility) ** 2
                portfolio_volatility = np.sqrt(portfolio_volatility)
            
            # Calculate concentration risk (Herfindahl index)
            weights = [pos.get('position_value', 0) / total_portfolio_value for pos in positions]
            concentration_risk = sum(w ** 2 for w in weights)
            
            # Calculate maximum drawdown risk
            max_drawdown_risk = portfolio_volatility * 2.5  # Approximate 99% VaR
            
            return {
                'total_risk': total_portfolio_value * portfolio_volatility,
                'portfolio_volatility': portfolio_volatility,
                'max_drawdown_risk': max_drawdown_risk,
                'concentration_risk': concentration_risk,
                'portfolio_value': total_portfolio_value
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {
                'total_risk': 0,
                'portfolio_volatility': 0,
                'max_drawdown_risk': 0,
                'concentration_risk': 0
            }
    
    def update_account_size(self, new_account_size: float):
        """Update account size for position sizing calculations."""
        self.account_size = new_account_size
        logger.info(f"Account size updated to: {new_account_size}")
    
    def get_position_sizing_summary(self) -> Dict[str, float]:
        """Get summary of position sizing parameters."""
        return {
            'account_size': self.account_size,
            'max_position_size': self.max_position_size,
            'max_risk_per_trade': self.max_risk_per_trade,
            'volatility_lookback': self.volatility_lookback
        } 