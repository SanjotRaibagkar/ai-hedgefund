"""
EOD Strategy Manager for AI Hedge Fund.
Manages and coordinates all EOD momentum strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import json
import os

from .momentum_framework import MomentumStrategyFramework
from .long_momentum import LongMomentumStrategy
from .short_momentum import ShortMomentumStrategy
from .momentum_indicators import MomentumIndicators
from .position_sizing import PositionSizing
from .risk_management import RiskManager


class EODStrategyManager:
    """Manager for EOD momentum strategies."""
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 portfolio_value: float = 100000,
                 **kwargs):
        """
        Initialize EOD strategy manager.
        
        Args:
            config_path: Path to strategy configuration file
            portfolio_value: Initial portfolio value
            **kwargs: Additional configuration parameters
        """
        self.config_path = config_path or "config/eod_strategies.json"
        self.portfolio_value = portfolio_value
        self.strategies = {}
        self.frameworks = {}
        self.performance_history = []
        self.positions = {}
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize strategies and frameworks
        self._initialize_strategies()
        
        logger.info(f"EOD Strategy Manager initialized with portfolio value: {portfolio_value}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load strategy configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                # Create default configuration
                config = self._create_default_config()
                self._save_configuration(config)
                logger.info(f"Created default configuration at {self.config_path}")
                return config
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default strategy configuration."""
        return {
            "portfolio_settings": {
                "initial_value": self.portfolio_value,
                "max_positions": 10,
                "max_long_positions": 5,
                "max_short_positions": 5,
                "portfolio_risk_limit": 0.02,
                "correlation_threshold": 0.7
            },
            "strategies": {
                "long_momentum": {
                    "enabled": True,
                    "min_signal_strength": 0.3,
                    "min_momentum_score": 20.0,
                    "min_volume_ratio": 1.5,
                    "max_holding_period": 20
                },
                "short_momentum": {
                    "enabled": True,
                    "min_signal_strength": 0.3,
                    "min_momentum_score": 20.0,
                    "min_volume_ratio": 1.5,
                    "max_holding_period": 20
                }
            },
            "position_sizing": {
                "method": "adaptive",
                "max_position_size": 0.1,
                "min_position_size": 0.01,
                "kelly_fraction": 0.25
            },
            "risk_management": {
                "max_portfolio_risk": 0.02,
                "max_position_risk": 0.01,
                "max_drawdown": 0.15,
                "max_positions": 10,
                "stop_loss_method": "adaptive",
                "take_profit_method": "fixed_ratio",
                "risk_reward_ratio": 2.0
            },
            "execution": {
                "simulation_mode": True,
                "slippage": 0.001,
                "commission": 0.001,
                "min_trade_size": 1000
            }
        }
    
    def _save_configuration(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _initialize_strategies(self) -> None:
        """Initialize individual strategies and frameworks."""
        try:
            # Initialize individual strategies
            if self.config['strategies']['long_momentum']['enabled']:
                self.strategies['long_momentum'] = LongMomentumStrategy(
                    **self.config['strategies']['long_momentum']
                )
                logger.info("Long momentum strategy initialized")
            
            if self.config['strategies']['short_momentum']['enabled']:
                self.strategies['short_momentum'] = ShortMomentumStrategy(
                    **self.config['strategies']['short_momentum']
                )
                logger.info("Short momentum strategy initialized")
            
            # Initialize momentum framework
            self.frameworks['momentum_framework'] = MomentumStrategyFramework(
                portfolio_value=self.portfolio_value,
                **self.config['portfolio_settings'],
                long_strategy=self.config['strategies']['long_momentum'],
                short_strategy=self.config['strategies']['short_momentum']
            )
            logger.info("Momentum framework initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
    
    def run_daily_analysis(self, 
                          universe_data: Dict[str, pd.DataFrame],
                          market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run daily analysis for the entire universe.
        
        Args:
            universe_data: Dictionary of ticker -> DataFrame mappings
            market_data: Optional market-wide data
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Running daily analysis for {len(universe_data)} tickers")
            
            results = {
                'timestamp': datetime.now(),
                'universe_size': len(universe_data),
                'framework_analysis': {},
                'individual_analysis': {},
                'recommendations': {},
                'performance_metrics': {}
            }
            
            # Run framework analysis
            if 'momentum_framework' in self.frameworks:
                framework_results = self.frameworks['momentum_framework'].analyze_universe(universe_data)
                results['framework_analysis'] = framework_results
                
                # Generate recommendations
                recommendations = self.frameworks['momentum_framework'].generate_trade_recommendations(framework_results)
                results['recommendations'] = recommendations
            
            # Run individual strategy analysis
            for strategy_name, strategy in self.strategies.items():
                strategy_results = {}
                for ticker, df in universe_data.items():
                    try:
                        analysis = strategy.analyze_stock(df, ticker)
                        strategy_results[ticker] = analysis
                    except Exception as e:
                        logger.error(f"Strategy {strategy_name} failed for {ticker}: {e}")
                        strategy_results[ticker] = {
                            'ticker': ticker,
                            'signal': 'error',
                            'confidence': 0,
                            'reason': f'Analysis failed: {e}'
                        }
                
                results['individual_analysis'][strategy_name] = strategy_results
            
            # Calculate performance metrics
            results['performance_metrics'] = self._calculate_performance_metrics()
            
            # Store results in history
            self.performance_history.append(results)
            
            logger.info("Daily analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Daily analysis failed: {e}")
            return {
                'error': f'Daily analysis failed: {e}',
                'timestamp': datetime.now()
            }
    
    def execute_trades(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade recommendations.
        
        Args:
            recommendations: Trade recommendations from analysis
            
        Returns:
            Dictionary with execution results
        """
        try:
            logger.info("Executing trade recommendations")
            
            execution_results = {
                'timestamp': datetime.now(),
                'executed_trades': [],
                'failed_trades': [],
                'portfolio_updates': {},
                'execution_summary': {}
            }
            
            # Execute exits first
            for exit_rec in recommendations.get('exit_positions', []):
                result = self._execute_exit_trade(exit_rec)
                if result['success']:
                    execution_results['executed_trades'].append(result)
                else:
                    execution_results['failed_trades'].append(result)
            
            # Execute new positions
            for new_pos_rec in recommendations.get('new_positions', []):
                result = self._execute_new_position_trade(new_pos_rec)
                if result['success']:
                    execution_results['executed_trades'].append(result)
                else:
                    execution_results['failed_trades'].append(result)
            
            # Update portfolio
            execution_results['portfolio_updates'] = self._update_portfolio_state()
            
            # Generate execution summary
            execution_results['execution_summary'] = self._generate_execution_summary(execution_results)
            
            logger.info(f"Trade execution completed: {len(execution_results['executed_trades'])} successful, {len(execution_results['failed_trades'])} failed")
            return execution_results
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                'error': f'Trade execution failed: {e}',
                'timestamp': datetime.now()
            }
    
    def get_strategy_performance(self, 
                               strategy_name: Optional[str] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get performance metrics for strategies.
        
        Args:
            strategy_name: Specific strategy name (None for all)
            start_date: Start date for performance calculation
            end_date: End date for performance calculation
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.performance_history:
                return {'error': 'No performance history available'}
            
            # Filter history by date range
            filtered_history = self.performance_history
            if start_date:
                filtered_history = [h for h in filtered_history if h['timestamp'] >= start_date]
            if end_date:
                filtered_history = [h for h in filtered_history if h['timestamp'] <= end_date]
            
            if not filtered_history:
                return {'error': 'No data in specified date range'}
            
            performance = {
                'period': {
                    'start': filtered_history[0]['timestamp'],
                    'end': filtered_history[-1]['timestamp']
                },
                'total_analyses': len(filtered_history),
                'strategies': {}
            }
            
            # Calculate performance for each strategy
            if strategy_name:
                strategies_to_analyze = [strategy_name] if strategy_name in self.strategies else []
            else:
                strategies_to_analyze = list(self.strategies.keys())
            
            for strategy in strategies_to_analyze:
                strategy_performance = self._calculate_strategy_performance(strategy, filtered_history)
                performance['strategies'][strategy] = strategy_performance
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {'error': f'Performance calculation failed: {e}'}
    
    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """
        Update strategy configuration.
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update configuration
            self.config.update(new_config)
            
            # Save to file
            self._save_configuration(self.config)
            
            # Reinitialize strategies with new configuration
            self._initialize_strategies()
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False
    
    def _execute_exit_trade(self, exit_rec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute exit trade."""
        try:
            ticker = exit_rec['ticker']
            position = self.positions.get(ticker)
            
            if not position:
                return {
                    'success': False,
                    'reason': 'Position not found',
                    'ticker': ticker
                }
            
            # Apply slippage and commission
            exit_price = exit_rec.get('exit_price', position.get('entry_price', 100))
            slippage = self.config['execution']['slippage']
            commission = self.config['execution']['commission']
            
            # Adjust exit price for slippage
            if position.get('type') == 'long':
                adjusted_exit_price = exit_price * (1 - slippage)
            else:
                adjusted_exit_price = exit_price * (1 + slippage)
            
            # Calculate P&L
            entry_price = position.get('entry_price', 100)
            if position.get('type') == 'long':
                pnl = (adjusted_exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - adjusted_exit_price) / entry_price
            
            # Apply commission
            pnl -= commission * 2  # Entry and exit commission
            
            # Update portfolio
            position_value = position.get('value', 0)
            self.portfolio_value += position_value * pnl
            del self.positions[ticker]
            
            return {
                'success': True,
                'ticker': ticker,
                'action': 'exit',
                'exit_price': adjusted_exit_price,
                'pnl': pnl,
                'reason': exit_rec.get('reason', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Exit trade execution failed: {e}")
            return {
                'success': False,
                'reason': f'Execution failed: {e}',
                'ticker': exit_rec.get('ticker', 'UNKNOWN')
            }
    
    def _execute_new_position_trade(self, new_pos_rec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute new position trade."""
        try:
            ticker = new_pos_rec['ticker']
            entry_price = new_pos_rec.get('entry_price', 100)
            position_size = new_pos_rec.get('position_size', 10000)
            
            # Check minimum trade size
            min_trade_size = self.config['execution']['min_trade_size']
            if position_size < min_trade_size:
                return {
                    'success': False,
                    'reason': f'Position size {position_size} below minimum {min_trade_size}',
                    'ticker': ticker
                }
            
            # Apply slippage and commission
            slippage = self.config['execution']['slippage']
            commission = self.config['execution']['commission']
            
            # Adjust entry price for slippage
            if new_pos_rec['position_type'] == 'long':
                adjusted_entry_price = entry_price * (1 + slippage)
            else:
                adjusted_entry_price = entry_price * (1 - slippage)
            
            # Calculate total cost including commission
            total_cost = position_size * (1 + commission)
            
            # Check if we have enough cash
            if total_cost > self.portfolio_value:
                return {
                    'success': False,
                    'reason': 'Insufficient portfolio value',
                    'ticker': ticker
                }
            
            # Execute trade
            self.portfolio_value -= total_cost
            self.positions[ticker] = {
                'ticker': ticker,
                'type': new_pos_rec['position_type'],
                'entry_price': adjusted_entry_price,
                'current_price': adjusted_entry_price,
                'value': position_size,
                'stop_loss': new_pos_rec.get('stop_loss'),
                'take_profit': new_pos_rec.get('take_profit'),
                'entry_date': datetime.now(),
                'confidence': new_pos_rec.get('confidence', 0)
            }
            
            return {
                'success': True,
                'ticker': ticker,
                'action': 'enter',
                'entry_price': adjusted_entry_price,
                'position_size': position_size,
                'position_type': new_pos_rec['position_type']
            }
            
        except Exception as e:
            logger.error(f"New position trade execution failed: {e}")
            return {
                'success': False,
                'reason': f'Execution failed: {e}',
                'ticker': new_pos_rec.get('ticker', 'UNKNOWN')
            }
    
    def _update_portfolio_state(self) -> Dict[str, Any]:
        """Update portfolio state after trades."""
        try:
            total_position_value = sum(p.get('value', 0) for p in self.positions.values())
            
            return {
                'portfolio_value': self.portfolio_value,
                'total_positions': len(self.positions),
                'exposure_ratio': total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0,
                'long_positions': len([p for p in self.positions.values() if p.get('type') == 'long']),
                'short_positions': len([p for p in self.positions.values() if p.get('type') == 'short'])
            }
            
        except Exception as e:
            logger.error(f"Portfolio state update failed: {e}")
            return {}
    
    def _generate_execution_summary(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution summary."""
        try:
            executed_trades = execution_results.get('executed_trades', [])
            failed_trades = execution_results.get('failed_trades', [])
            
            summary = {
                'total_trades': len(executed_trades) + len(failed_trades),
                'successful_trades': len(executed_trades),
                'failed_trades': len(failed_trades),
                'success_rate': len(executed_trades) / (len(executed_trades) + len(failed_trades)) if (len(executed_trades) + len(failed_trades)) > 0 else 0,
                'total_pnl': sum(t.get('pnl', 0) for t in executed_trades if t.get('action') == 'exit'),
                'portfolio_value': self.portfolio_value
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Execution summary generation failed: {e}")
            return {}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics."""
        try:
            metrics = {
                'portfolio_value': self.portfolio_value,
                'total_positions': len(self.positions),
                'long_positions': len([p for p in self.positions.values() if p.get('type') == 'long']),
                'short_positions': len([p for p in self.positions.values() if p.get('type') == 'short']),
                'exposure_ratio': sum(p.get('value', 0) for p in self.positions.values()) / self.portfolio_value if self.portfolio_value > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    def _calculate_strategy_performance(self, strategy_name: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance for a specific strategy."""
        try:
            strategy_performance = {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'error_signals': 0,
                'avg_confidence': 0.0
            }
            
            total_confidence = 0.0
            confidence_count = 0
            
            for analysis in history:
                if strategy_name in analysis.get('individual_analysis', {}):
                    strategy_analysis = analysis['individual_analysis'][strategy_name]
                    
                    for ticker, result in strategy_analysis.items():
                        strategy_performance['total_signals'] += 1
                        
                        signal = result.get('signal', 'error')
                        if signal == 'buy':
                            strategy_performance['buy_signals'] += 1
                        elif signal == 'sell':
                            strategy_performance['sell_signals'] += 1
                        elif signal == 'hold':
                            strategy_performance['hold_signals'] += 1
                        else:
                            strategy_performance['error_signals'] += 1
                        
                        confidence = result.get('confidence', 0)
                        if confidence > 0:
                            total_confidence += confidence
                            confidence_count += 1
            
            if confidence_count > 0:
                strategy_performance['avg_confidence'] = total_confidence / confidence_count
            
            return strategy_performance
            
        except Exception as e:
            logger.error(f"Strategy performance calculation failed for {strategy_name}: {e}")
            return {'error': f'Performance calculation failed: {e}'}
    
    def get_manager_summary(self) -> Dict[str, Any]:
        """Get comprehensive manager summary."""
        return {
            'manager_info': {
                'portfolio_value': self.portfolio_value,
                'total_positions': len(self.positions),
                'active_strategies': list(self.strategies.keys()),
                'active_frameworks': list(self.frameworks.keys())
            },
            'configuration': self.config,
            'performance_history_length': len(self.performance_history),
            'strategy_summaries': {
                name: strategy.get_strategy_summary() 
                for name, strategy in self.strategies.items()
            },
            'framework_summaries': {
                name: framework.get_framework_summary() 
                for name, framework in self.frameworks.items()
            }
        } 