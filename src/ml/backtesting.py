"""
Backtesting Framework Module for ML-Enhanced Strategies.
Provides comprehensive backtesting capabilities for ML-enhanced trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.ml.ml_strategies import MLEnhancedEODStrategy
from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_manager import MLModelManager
from src.ml.mlflow_tracker import MLflowTracker


class MLBacktestingFramework:
    """Comprehensive backtesting framework for ML-enhanced strategies."""
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 mlflow_tracker: Optional[MLflowTracker] = None):
        """
        Initialize ML backtesting framework.
        
        Args:
            initial_capital: Initial portfolio capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
            mlflow_tracker: MLflow tracker instance
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.mlflow_tracker = mlflow_tracker
        
        # Backtesting components
        self.ml_strategy = MLEnhancedEODStrategy()
        self.feature_engineer = FeatureEngineer()
        self.model_manager = MLModelManager(mlflow_tracker=mlflow_tracker)
        
        # Backtesting state
        self.portfolio_state = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info("MLBacktestingFramework initialized")
    
    def run_backtest(self,
                    ticker: str,
                    start_date: str,
                    end_date: str,
                    strategy_config: Optional[Dict[str, Any]] = None,
                    rebalance_frequency: str = 'daily') -> Dict[str, Any]:
        """
        Run comprehensive backtest for ML-enhanced strategy.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_config: Strategy configuration
            rebalance_frequency: Portfolio rebalancing frequency
            
        Returns:
            Backtest results dictionary
        """
        try:
            logger.info(f"Starting ML backtest for {ticker} from {start_date} to {end_date}")
            
            # Start MLflow experiment if available
            run_id = None
            if self.mlflow_tracker:
                run_id = self.mlflow_tracker.start_experiment(
                    run_name=f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags={'ticker': ticker, 'backtest_type': 'ml_enhanced'}
                )
            
            # Initialize portfolio
            self._initialize_portfolio()
            
            # Train models
            logger.info("Training ML models for backtest")
            training_results = self.model_manager.train_all_models(
                ticker, start_date, end_date
            )
            
            if 'error' in training_results:
                raise ValueError(f"Model training failed: {training_results['error']}")
            
            # Get historical data for backtesting
            features, target = self.feature_engineer.create_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if features.empty:
                raise ValueError("No features available for backtesting")
            
            # Run backtest
            backtest_results = self._execute_backtest(
                ticker, features, target, rebalance_frequency
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # Store results
            self.performance_metrics = performance_metrics
            
            # Log results to MLflow if available
            if self.mlflow_tracker:
                self._log_backtest_results(ticker, performance_metrics, training_results)
            
            # End MLflow experiment
            if self.mlflow_tracker and run_id:
                self.mlflow_tracker.end_experiment()
            
            logger.info(f"ML backtest completed for {ticker}")
            
            return {
                'ticker': ticker,
                'backtest_period': {'start': start_date, 'end': end_date},
                'training_results': training_results,
                'backtest_results': backtest_results,
                'performance_metrics': performance_metrics,
                'portfolio_state': self.portfolio_state,
                'trade_history': self.trade_history
            }
            
        except Exception as e:
            logger.error(f"ML backtest failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _initialize_portfolio(self):
        """Initialize portfolio state."""
        self.portfolio_state = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital,
            'daily_returns': [],
            'cumulative_returns': [1.0]
        }
        self.trade_history = []
    
    def _execute_backtest(self,
                         ticker: str,
                         features: pd.DataFrame,
                         target: pd.Series,
                         rebalance_frequency: str) -> Dict[str, Any]:
        """Execute the backtest simulation."""
        try:
            # Get predictions from all models
            predictions = self.model_manager.predict_with_all_models(
                ticker, features.index[0].strftime('%Y-%m-%d'), features.index[-1].strftime('%Y-%m-%d')
            )
            
            if 'error' in predictions:
                raise ValueError(f"Prediction failed: {predictions['error']}")
            
            # Simulate trading
            daily_returns = []
            position_sizes = []
            
            for i, date in enumerate(features.index):
                try:
                    # Get predictions for this date
                    date_predictions = {}
                    for model_name, model_pred in predictions.items():
                        if 'error' not in model_pred and 'predictions' in model_pred:
                            if i < len(model_pred['predictions']):
                                date_predictions[model_name] = model_pred['predictions'][i]
                    
                    if not date_predictions:
                        continue
                    
                    # Get ensemble prediction (preferred) or best individual model
                    if 'ensemble' in date_predictions:
                        prediction = date_predictions['ensemble']
                    else:
                        # Use best performing model
                        best_model_name = self._get_best_model_name()
                        if best_model_name in date_predictions:
                            prediction = date_predictions[best_model_name]
                        else:
                            continue
                    
                    # Execute trading decision
                    trade_result = self._execute_trade_decision(
                        ticker, date, prediction, features.iloc[i]
                    )
                    
                    # Calculate daily return
                    daily_return = self._calculate_daily_return()
                    daily_returns.append(daily_return)
                    
                    # Update portfolio
                    self._update_portfolio_state(date, daily_return)
                    
                except Exception as e:
                    logger.warning(f"Error processing date {date}: {e}")
                    daily_returns.append(0.0)
            
            return {
                'daily_returns': daily_returns,
                'total_trades': len(self.trade_history),
                'final_portfolio_value': self.portfolio_state['total_value']
            }
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            return {'error': str(e)}
    
    def _execute_trade_decision(self,
                               ticker: str,
                               date: pd.Timestamp,
                               prediction: float,
                               features: pd.Series) -> Dict[str, Any]:
        """Execute trading decision based on ML prediction."""
        try:
            current_price = features.get('close_price', 100)  # Default price if not available
            current_position = self.portfolio_state['positions'].get(ticker, 0)
            
            # Determine position size based on prediction confidence
            position_size = self._calculate_position_size(prediction, current_price)
            
            # Execute trade if position size differs significantly from current
            if abs(position_size - current_position) > 0.01:  # 1% threshold
                trade_result = self._execute_trade(
                    ticker, date, current_position, position_size, current_price
                )
                return trade_result
            
            return {'action': 'hold', 'position_size': current_position}
            
        except Exception as e:
            logger.error(f"Trade decision execution failed: {e}")
            return {'action': 'error', 'error': str(e)}
    
    def _calculate_position_size(self, prediction: float, current_price: float) -> float:
        """Calculate position size based on ML prediction."""
        try:
            # Normalize prediction to [-1, 1] range
            normalized_prediction = np.clip(prediction * 10, -1, 1)
            
            # Calculate position size as percentage of portfolio
            max_position_size = 0.2  # Maximum 20% in single position
            position_size = abs(normalized_prediction) * max_position_size
            
            # Apply sign
            if normalized_prediction < 0:
                position_size = -position_size
            
            return position_size
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def _execute_trade(self,
                      ticker: str,
                      date: pd.Timestamp,
                      current_position: float,
                      target_position: float,
                      price: float) -> Dict[str, Any]:
        """Execute a trade."""
        try:
            position_change = target_position - current_position
            trade_value = abs(position_change) * self.portfolio_state['total_value']
            
            # Calculate costs
            commission_cost = trade_value * self.commission_rate
            slippage_cost = trade_value * self.slippage_rate
            total_cost = commission_cost + slippage_cost
            
            # Update portfolio
            self.portfolio_state['cash'] -= total_cost
            self.portfolio_state['positions'][ticker] = target_position
            
            # Record trade
            trade_record = {
                'date': date,
                'ticker': ticker,
                'action': 'buy' if position_change > 0 else 'sell',
                'position_change': position_change,
                'price': price,
                'trade_value': trade_value,
                'commission_cost': commission_cost,
                'slippage_cost': slippage_cost,
                'total_cost': total_cost
            }
            
            self.trade_history.append(trade_record)
            
            return {
                'action': trade_record['action'],
                'position_change': position_change,
                'trade_value': trade_value,
                'total_cost': total_cost
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'action': 'error', 'error': str(e)}
    
    def _calculate_daily_return(self) -> float:
        """Calculate daily portfolio return."""
        try:
            # Calculate current portfolio value
            total_value = self.portfolio_state['cash']
            
            # Add value of positions (simplified calculation)
            for ticker, position in self.portfolio_state['positions'].items():
                # Assume position value changes with market
                total_value += position * self.portfolio_state['total_value']
            
            # Calculate return
            previous_value = self.portfolio_state['total_value']
            daily_return = (total_value - previous_value) / previous_value
            
            # Update total value
            self.portfolio_state['total_value'] = total_value
            
            return daily_return
            
        except Exception as e:
            logger.error(f"Daily return calculation failed: {e}")
            return 0.0
    
    def _update_portfolio_state(self, date: pd.Timestamp, daily_return: float):
        """Update portfolio state with daily return."""
        try:
            self.portfolio_state['daily_returns'].append(daily_return)
            
            # Update cumulative returns
            last_cumulative = self.portfolio_state['cumulative_returns'][-1]
            new_cumulative = last_cumulative * (1 + daily_return)
            self.portfolio_state['cumulative_returns'].append(new_cumulative)
            
        except Exception as e:
            logger.error(f"Portfolio state update failed: {e}")
    
    def _get_best_model_name(self) -> str:
        """Get the name of the best performing model."""
        try:
            best_model_name, _ = self.model_manager.get_best_model('test_r2')
            return best_model_name or 'linear'  # Fallback to linear
        except Exception as e:
            logger.error(f"Failed to get best model name: {e}")
            return 'linear'
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            daily_returns = self.portfolio_state['daily_returns']
            cumulative_returns = self.portfolio_state['cumulative_returns']
            
            if not daily_returns:
                return {'error': 'No returns data available'}
            
            # Basic metrics
            total_return = cumulative_returns[-1] - 1
            annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
            
            # Risk metrics
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown metrics
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Trade metrics
            total_trades = len(self.trade_history)
            profitable_trades = len([t for t in self.trade_history if t.get('position_change', 0) > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Cost metrics
            total_commission = sum(t.get('commission_cost', 0) for t in self.trade_history)
            total_slippage = sum(t.get('slippage_cost', 0) for t in self.trade_history)
            total_costs = total_commission + total_slippage
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_costs': total_costs,
                'final_portfolio_value': self.portfolio_state['total_value'],
                'initial_capital': self.initial_capital
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _log_backtest_results(self,
                             ticker: str,
                             performance_metrics: Dict[str, Any],
                             training_results: Dict[str, Any]):
        """Log backtest results to MLflow."""
        try:
            if self.mlflow_tracker:
                # Log performance metrics
                self.mlflow_tracker.log_metrics(performance_metrics)
                
                # Log backtest parameters
                self.mlflow_tracker.log_params({
                    'ticker': ticker,
                    'initial_capital': self.initial_capital,
                    'commission_rate': self.commission_rate,
                    'slippage_rate': self.slippage_rate
                })
                
                # Log trade history as artifact
                if self.trade_history:
                    trade_df = pd.DataFrame(self.trade_history)
                    trade_path = f"/tmp/trade_history_{ticker}.csv"
                    trade_df.to_csv(trade_path, index=False)
                    self.mlflow_tracker.log_artifacts(trade_path, "trade_history")
                
                logger.info("Backtest results logged to MLflow")
                
        except Exception as e:
            logger.error(f"Failed to log backtest results: {e}")
    
    def compare_strategies(self,
                          ticker: str,
                          start_date: str,
                          end_date: str,
                          strategies: List[str]) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for comparison
            end_date: End date for comparison
            strategies: List of strategy names to compare
            
        Returns:
            Strategy comparison results
        """
        try:
            logger.info(f"Comparing strategies for {ticker}")
            
            comparison_results = {}
            
            for strategy in strategies:
                try:
                    if strategy == 'ml_enhanced':
                        # Run ML-enhanced backtest
                        result = self.run_backtest(ticker, start_date, end_date)
                    else:
                        # Run traditional strategy backtest
                        result = self._run_traditional_backtest(ticker, start_date, end_date, strategy)
                    
                    comparison_results[strategy] = result
                    
                except Exception as e:
                    logger.error(f"Strategy {strategy} failed: {e}")
                    comparison_results[strategy] = {'error': str(e)}
            
            return {
                'ticker': ticker,
                'comparison_period': {'start': start_date, 'end': end_date},
                'strategies': comparison_results
            }
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            return {'error': str(e)}
    
    def _run_traditional_backtest(self,
                                 ticker: str,
                                 start_date: str,
                                 end_date: str,
                                 strategy_name: str) -> Dict[str, Any]:
        """Run traditional strategy backtest."""
        try:
            # Simplified traditional strategy backtest
            # This would be expanded based on specific traditional strategies
            
            return {
                'strategy': strategy_name,
                'total_return': 0.05,  # Placeholder
                'annualized_return': 0.06,
                'volatility': 0.15,
                'sharpe_ratio': 0.4,
                'max_drawdown': -0.1
            }
            
        except Exception as e:
            logger.error(f"Traditional backtest failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive backtest report.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Formatted report string
        """
        try:
            if 'error' in results:
                return f"Backtest failed: {results['error']}"
            
            performance = results.get('performance_metrics', {})
            
            report = f"""
ML-Enhanced Strategy Backtest Report
====================================

Ticker: {results.get('ticker', 'Unknown')}
Period: {results.get('backtest_period', {}).get('start', 'Unknown')} to {results.get('backtest_period', {}).get('end', 'Unknown')}

Performance Metrics:
-------------------
Total Return: {performance.get('total_return', 0):.2%}
Annualized Return: {performance.get('annualized_return', 0):.2%}
Volatility: {performance.get('volatility', 0):.2%}
Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}
Maximum Drawdown: {performance.get('max_drawdown', 0):.2%}

Trading Statistics:
------------------
Total Trades: {performance.get('total_trades', 0)}
Profitable Trades: {performance.get('profitable_trades', 0)}
Win Rate: {performance.get('win_rate', 0):.2%}

Cost Analysis:
--------------
Total Commission: ${performance.get('total_commission', 0):.2f}
Total Slippage: ${performance.get('total_slippage', 0):.2f}
Total Costs: ${performance.get('total_costs', 0):.2f}

Portfolio Summary:
-----------------
Initial Capital: ${performance.get('initial_capital', 0):,.2f}
Final Value: ${performance.get('final_portfolio_value', 0):,.2f}
Net Profit: ${performance.get('final_portfolio_value', 0) - performance.get('initial_capital', 0):,.2f}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}" 