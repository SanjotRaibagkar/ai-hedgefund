"""
Risk Management for EOD Momentum Strategies.
Handles stop losses, take profits, and risk controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class RiskManager:
    """Handles risk management for EOD momentum strategies."""
    
    def __init__(self,
                 max_portfolio_risk: float = 0.05,  # 5% max portfolio risk
                 max_position_risk: float = 0.02,   # 2% max position risk
                 max_drawdown: float = 0.15,        # 15% max drawdown
                 correlation_threshold: float = 0.7, # Max correlation between positions
                 max_positions: int = 10):          # Max concurrent positions
        """
        Initialize risk manager.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk as fraction
            max_position_risk: Maximum position risk as fraction
            max_drawdown: Maximum allowed drawdown
            correlation_threshold: Maximum correlation between positions
            max_positions: Maximum number of concurrent positions
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_drawdown = max_drawdown
        self.correlation_threshold = correlation_threshold
        self.max_positions = max_positions
        
        # Risk management methods
        self.risk_methods = {
            'atr': self._atr_based_stops,
            'percentage': self._percentage_based_stops,
            'support_resistance': self._support_resistance_stops,
            'volatility': self._volatility_based_stops,
            'adaptive': self._adaptive_risk_management
        }
    
    def calculate_stop_loss(self, 
                          ticker: str,
                          entry_price: float,
                          position_type: str,  # 'long' or 'short'
                          method: str = 'adaptive',
                          atr: float = None,
                          volatility: float = None,
                          support_level: float = None,
                          resistance_level: float = None,
                          **kwargs) -> Dict[str, float]:
        """
        Calculate stop loss level for a position.
        
        Args:
            ticker: Stock ticker symbol
            entry_price: Entry price for the position
            position_type: 'long' or 'short'
            method: Stop loss calculation method
            atr: Average True Range value
            volatility: Stock volatility
            support_level: Support level for long positions
            resistance_level: Resistance level for short positions
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with stop loss details
        """
        try:
            if method not in self.risk_methods:
                logger.warning(f"Unknown risk method: {method}. Using adaptive.")
                method = 'adaptive'
            
            risk_func = self.risk_methods[method]
            stop_loss = risk_func(
                ticker=ticker,
                entry_price=entry_price,
                position_type=position_type,
                atr=atr,
                volatility=volatility,
                support_level=support_level,
                resistance_level=resistance_level,
                **kwargs
            )
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Failed to calculate stop loss: {e}")
            return self._get_default_stop_loss(entry_price, position_type)
    
    def calculate_take_profit(self, 
                            ticker: str,
                            entry_price: float,
                            stop_loss: float,
                            position_type: str,
                            risk_reward_ratio: float = 2.0,
                            method: str = 'fixed_ratio',
                            **kwargs) -> Dict[str, float]:
        """
        Calculate take profit level for a position.
        
        Args:
            ticker: Stock ticker symbol
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            position_type: 'long' or 'short'
            risk_reward_ratio: Risk to reward ratio
            method: Take profit calculation method
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with take profit details
        """
        try:
            if method == 'fixed_ratio':
                return self._fixed_ratio_take_profit(
                    entry_price, stop_loss, position_type, risk_reward_ratio
                )
            elif method == 'support_resistance':
                return self._support_resistance_take_profit(
                    ticker, entry_price, stop_loss, position_type, **kwargs
                )
            elif method == 'volatility':
                return self._volatility_based_take_profit(
                    entry_price, stop_loss, position_type, **kwargs
                )
            else:
                return self._fixed_ratio_take_profit(
                    entry_price, stop_loss, position_type, risk_reward_ratio
                )
                
        except Exception as e:
            logger.error(f"Failed to calculate take profit: {e}")
            return self._get_default_take_profit(entry_price, stop_loss, position_type)
    
    def _atr_based_stops(self, 
                        ticker: str,
                        entry_price: float,
                        position_type: str,
                        atr: float = None,
                        atr_multiplier: float = 2.0,
                        **kwargs) -> Dict[str, float]:
        """ATR-based stop loss calculation."""
        try:
            if atr is None:
                atr = entry_price * 0.02  # Default 2% ATR
            
            stop_distance = atr * atr_multiplier
            
            if position_type == 'long':
                stop_loss = entry_price - stop_distance
            else:  # short
                stop_loss = entry_price + stop_distance
            
            risk_amount = abs(entry_price - stop_loss)
            risk_percentage = (risk_amount / entry_price) * 100
            
            return {
                'stop_loss': stop_loss,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'method': 'atr',
                'atr_multiplier': atr_multiplier
            }
            
        except Exception as e:
            logger.error(f"ATR-based stops calculation failed: {e}")
            return self._get_default_stop_loss(entry_price, position_type)
    
    def _percentage_based_stops(self, 
                               ticker: str,
                               entry_price: float,
                               position_type: str,
                               stop_percentage: float = 0.05,
                               **kwargs) -> Dict[str, float]:
        """Percentage-based stop loss calculation."""
        try:
            stop_distance = entry_price * stop_percentage
            
            if position_type == 'long':
                stop_loss = entry_price - stop_distance
            else:  # short
                stop_loss = entry_price + stop_distance
            
            risk_amount = abs(entry_price - stop_loss)
            risk_percentage = (risk_amount / entry_price) * 100
            
            return {
                'stop_loss': stop_loss,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'method': 'percentage',
                'stop_percentage': stop_percentage
            }
            
        except Exception as e:
            logger.error(f"Percentage-based stops calculation failed: {e}")
            return self._get_default_stop_loss(entry_price, position_type)
    
    def _support_resistance_stops(self, 
                                 ticker: str,
                                 entry_price: float,
                                 position_type: str,
                                 support_level: float = None,
                                 resistance_level: float = None,
                                 buffer_percentage: float = 0.01,
                                 **kwargs) -> Dict[str, float]:
        """Support/Resistance-based stop loss calculation."""
        try:
            if position_type == 'long':
                if support_level is None:
                    support_level = entry_price * 0.95  # Default 5% below entry
                
                stop_loss = support_level * (1 - buffer_percentage)
            else:  # short
                if resistance_level is None:
                    resistance_level = entry_price * 1.05  # Default 5% above entry
                
                stop_loss = resistance_level * (1 + buffer_percentage)
            
            risk_amount = abs(entry_price - stop_loss)
            risk_percentage = (risk_amount / entry_price) * 100
            
            return {
                'stop_loss': stop_loss,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'method': 'support_resistance',
                'support_level': support_level,
                'resistance_level': resistance_level
            }
            
        except Exception as e:
            logger.error(f"Support/Resistance stops calculation failed: {e}")
            return self._get_default_stop_loss(entry_price, position_type)
    
    def _volatility_based_stops(self, 
                               ticker: str,
                               entry_price: float,
                               position_type: str,
                               volatility: float = None,
                               volatility_multiplier: float = 1.5,
                               **kwargs) -> Dict[str, float]:
        """Volatility-based stop loss calculation."""
        try:
            if volatility is None:
                volatility = 0.2  # Default 20% volatility
            
            stop_distance = entry_price * volatility * volatility_multiplier
            
            if position_type == 'long':
                stop_loss = entry_price - stop_distance
            else:  # short
                stop_loss = entry_price + stop_distance
            
            risk_amount = abs(entry_price - stop_loss)
            risk_percentage = (risk_amount / entry_price) * 100
            
            return {
                'stop_loss': stop_loss,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'method': 'volatility',
                'volatility_multiplier': volatility_multiplier
            }
            
        except Exception as e:
            logger.error(f"Volatility-based stops calculation failed: {e}")
            return self._get_default_stop_loss(entry_price, position_type)
    
    def _adaptive_risk_management(self, 
                                 ticker: str,
                                 entry_price: float,
                                 position_type: str,
                                 atr: float = None,
                                 volatility: float = None,
                                 market_conditions: str = 'normal',
                                 **kwargs) -> Dict[str, float]:
        """Adaptive risk management combining multiple factors."""
        try:
            # Base stop distance
            base_stop_percentage = 0.05  # 5% base
            
            # Adjust based on volatility
            if volatility is None:
                volatility = 0.2
            volatility_adjustment = volatility / 0.2  # Normalize to 20% volatility
            
            # Adjust based on market conditions
            market_adjustments = {
                'bull': 0.8,      # Tighter stops in bull market
                'normal': 1.0,    # Normal stops
                'bear': 1.2,      # Wider stops in bear market
                'volatile': 1.5   # Much wider stops in volatile market
            }
            market_adjustment = market_adjustments.get(market_conditions, 1.0)
            
            # Calculate final stop distance
            stop_percentage = base_stop_percentage * volatility_adjustment * market_adjustment
            stop_distance = entry_price * stop_percentage
            
            if position_type == 'long':
                stop_loss = entry_price - stop_distance
            else:  # short
                stop_loss = entry_price + stop_distance
            
            risk_amount = abs(entry_price - stop_loss)
            risk_percentage = (risk_amount / entry_price) * 100
            
            return {
                'stop_loss': stop_loss,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'method': 'adaptive',
                'volatility_adjustment': volatility_adjustment,
                'market_adjustment': market_adjustment,
                'stop_percentage': stop_percentage
            }
            
        except Exception as e:
            logger.error(f"Adaptive risk management failed: {e}")
            return self._get_default_stop_loss(entry_price, position_type)
    
    def _fixed_ratio_take_profit(self, 
                                entry_price: float,
                                stop_loss: float,
                                position_type: str,
                                risk_reward_ratio: float = 2.0) -> Dict[str, float]:
        """Fixed risk-reward ratio take profit calculation."""
        try:
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = risk_amount * risk_reward_ratio
            
            if position_type == 'long':
                take_profit = entry_price + reward_amount
            else:  # short
                take_profit = entry_price - reward_amount
            
            reward_percentage = (reward_amount / entry_price) * 100
            
            return {
                'take_profit': take_profit,
                'reward_amount': reward_amount,
                'reward_percentage': reward_percentage,
                'risk_reward_ratio': risk_reward_ratio,
                'method': 'fixed_ratio'
            }
            
        except Exception as e:
            logger.error(f"Fixed ratio take profit calculation failed: {e}")
            return self._get_default_take_profit(entry_price, stop_loss, position_type)
    
    def _support_resistance_take_profit(self, 
                                       ticker: str,
                                       entry_price: float,
                                       stop_loss: float,
                                       position_type: str,
                                       resistance_level: float = None,
                                       support_level: float = None,
                                       **kwargs) -> Dict[str, float]:
        """Support/Resistance-based take profit calculation."""
        try:
            if position_type == 'long':
                if resistance_level is None:
                    resistance_level = entry_price * 1.10  # Default 10% above entry
                take_profit = resistance_level
            else:  # short
                if support_level is None:
                    support_level = entry_price * 0.90  # Default 10% below entry
                take_profit = support_level
            
            reward_amount = abs(entry_price - take_profit)
            reward_percentage = (reward_amount / entry_price) * 100
            risk_amount = abs(entry_price - stop_loss)
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                'take_profit': take_profit,
                'reward_amount': reward_amount,
                'reward_percentage': reward_percentage,
                'risk_reward_ratio': risk_reward_ratio,
                'method': 'support_resistance'
            }
            
        except Exception as e:
            logger.error(f"Support/Resistance take profit calculation failed: {e}")
            return self._get_default_take_profit(entry_price, stop_loss, position_type)
    
    def _volatility_based_take_profit(self, 
                                     entry_price: float,
                                     stop_loss: float,
                                     position_type: str,
                                     volatility: float = None,
                                     **kwargs) -> Dict[str, float]:
        """Volatility-based take profit calculation."""
        try:
            if volatility is None:
                volatility = 0.2
            
            # Use volatility to determine take profit distance
            take_profit_distance = entry_price * volatility * 2  # 2x volatility
            
            if position_type == 'long':
                take_profit = entry_price + take_profit_distance
            else:  # short
                take_profit = entry_price - take_profit_distance
            
            reward_amount = abs(entry_price - take_profit)
            reward_percentage = (reward_amount / entry_price) * 100
            risk_amount = abs(entry_price - stop_loss)
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                'take_profit': take_profit,
                'reward_amount': reward_amount,
                'reward_percentage': reward_percentage,
                'risk_reward_ratio': risk_reward_ratio,
                'method': 'volatility'
            }
            
        except Exception as e:
            logger.error(f"Volatility-based take profit calculation failed: {e}")
            return self._get_default_take_profit(entry_price, stop_loss, position_type)
    
    def _get_default_stop_loss(self, entry_price: float, position_type: str) -> Dict[str, float]:
        """Get default stop loss when calculation fails."""
        try:
            default_stop_percentage = 0.05  # 5% default
            stop_distance = entry_price * default_stop_percentage
            
            if position_type == 'long':
                stop_loss = entry_price - stop_distance
            else:  # short
                stop_loss = entry_price + stop_distance
            
            return {
                'stop_loss': stop_loss,
                'risk_amount': stop_distance,
                'risk_percentage': default_stop_percentage * 100,
                'method': 'default'
            }
            
        except Exception as e:
            logger.error(f"Failed to get default stop loss: {e}")
            return {
                'stop_loss': entry_price * 0.95 if position_type == 'long' else entry_price * 1.05,
                'risk_amount': entry_price * 0.05,
                'risk_percentage': 5.0,
                'method': 'fallback'
            }
    
    def _get_default_take_profit(self, entry_price: float, stop_loss: float, position_type: str) -> Dict[str, float]:
        """Get default take profit when calculation fails."""
        try:
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = risk_amount * 2  # 2:1 risk-reward ratio
            
            if position_type == 'long':
                take_profit = entry_price + reward_amount
            else:  # short
                take_profit = entry_price - reward_amount
            
            return {
                'take_profit': take_profit,
                'reward_amount': reward_amount,
                'reward_percentage': (reward_amount / entry_price) * 100,
                'risk_reward_ratio': 2.0,
                'method': 'default'
            }
            
        except Exception as e:
            logger.error(f"Failed to get default take profit: {e}")
            return {
                'take_profit': entry_price * 1.10 if position_type == 'long' else entry_price * 0.90,
                'reward_amount': entry_price * 0.10,
                'reward_percentage': 10.0,
                'risk_reward_ratio': 2.0,
                'method': 'fallback'
            }
    
    def check_portfolio_risk(self, 
                           positions: List[Dict[str, any]],
                           account_value: float) -> Dict[str, any]:
        """
        Check portfolio risk levels.
        
        Args:
            positions: List of current positions
            account_value: Current account value
            
        Returns:
            Dictionary with risk assessment
        """
        try:
            total_position_value = sum(pos.get('position_value', 0) for pos in positions)
            total_risk = sum(pos.get('risk_amount', 0) * pos.get('shares', 0) for pos in positions)
            
            portfolio_risk_ratio = total_risk / account_value if account_value > 0 else 0
            position_concentration = total_position_value / account_value if account_value > 0 else 0
            
            risk_assessment = {
                'total_positions': len(positions),
                'total_position_value': total_position_value,
                'total_risk': total_risk,
                'portfolio_risk_ratio': portfolio_risk_ratio,
                'position_concentration': position_concentration,
                'risk_status': 'normal',
                'warnings': []
            }
            
            # Check risk limits
            if portfolio_risk_ratio > self.max_portfolio_risk:
                risk_assessment['risk_status'] = 'high'
                risk_assessment['warnings'].append(f"Portfolio risk ({portfolio_risk_ratio:.2%}) exceeds limit ({self.max_portfolio_risk:.2%})")
            
            if position_concentration > 0.8:  # 80% concentration
                risk_assessment['warnings'].append(f"High position concentration ({position_concentration:.2%})")
            
            if len(positions) > self.max_positions:
                risk_assessment['warnings'].append(f"Too many positions ({len(positions)} > {self.max_positions})")
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Failed to check portfolio risk: {e}")
            return {
                'risk_status': 'error',
                'warnings': [f"Risk check failed: {e}"]
            }
    
    def should_exit_position(self, 
                           position: Dict[str, any],
                           current_price: float,
                           current_drawdown: float = None) -> Dict[str, any]:
        """
        Check if position should be exited based on risk rules.
        
        Args:
            position: Position dictionary
            current_price: Current stock price
            current_drawdown: Current portfolio drawdown
            
        Returns:
            Dictionary with exit decision
        """
        try:
            exit_decision = {
                'should_exit': False,
                'reason': None,
                'exit_type': None
            }
            
            # Check stop loss
            stop_loss = position.get('stop_loss')
            if stop_loss is not None:
                if position.get('position_type') == 'long' and current_price <= stop_loss:
                    exit_decision['should_exit'] = True
                    exit_decision['reason'] = 'stop_loss'
                    exit_decision['exit_type'] = 'stop_loss'
                elif position.get('position_type') == 'short' and current_price >= stop_loss:
                    exit_decision['should_exit'] = True
                    exit_decision['reason'] = 'stop_loss'
                    exit_decision['exit_type'] = 'stop_loss'
            
            # Check take profit
            take_profit = position.get('take_profit')
            if take_profit is not None:
                if position.get('position_type') == 'long' and current_price >= take_profit:
                    exit_decision['should_exit'] = True
                    exit_decision['reason'] = 'take_profit'
                    exit_decision['exit_type'] = 'take_profit'
                elif position.get('position_type') == 'short' and current_price <= take_profit:
                    exit_decision['should_exit'] = True
                    exit_decision['reason'] = 'take_profit'
                    exit_decision['exit_type'] = 'take_profit'
            
            # Check drawdown limit
            if current_drawdown is not None and current_drawdown > self.max_drawdown:
                exit_decision['should_exit'] = True
                exit_decision['reason'] = 'max_drawdown'
                exit_decision['exit_type'] = 'risk_management'
            
            return exit_decision
            
        except Exception as e:
            logger.error(f"Failed to check exit conditions: {e}")
            return {
                'should_exit': False,
                'reason': 'error',
                'exit_type': None
            } 