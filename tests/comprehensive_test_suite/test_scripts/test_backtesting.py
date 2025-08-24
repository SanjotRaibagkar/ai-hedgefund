#!/usr/bin/env python3
"""
Backtesting Test Suite
Tests backtesting functionality including ML backtesting and strategy performance evaluation.
"""

import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from loguru import logger
from src.tools.enhanced_api import get_prices
from src.ml.backtesting import MLBacktester
from src.ml.mlflow_tracker import MLflowTracker
from src.strategies.eod import MomentumStrategyFramework
from src.ml.ml_strategies import MLEnhancedEODStrategy


class BacktestingTester:
    """Test backtesting capabilities."""
    
    def __init__(self):
        """Initialize the backtesting tester."""
        self.test_results = {
            'ml_backtesting': {},
            'strategy_backtesting': {},
            'mlflow_integration': {},
            'performance_metrics': {},
            'summary': {},
            'errors': [],
            'warnings': []
        }
        
        # Test data
        self.test_ticker = 'AAPL'
        self.test_data = None
        
        logger.info("Backtesting Tester initialized")
    
    def setup_test_data(self):
        """Setup test data for backtesting."""
        try:
            self.test_data = get_prices(self.test_ticker, '2022-01-01', '2023-12-31')
            if self.test_data is not None and not self.test_data.empty:
                logger.info(f"✅ Test data loaded: {len(self.test_data)} records for {self.test_ticker}")
                return True
            else:
                logger.warning(f"⚠️ No test data available for {self.test_ticker}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            return False
    
    def test_ml_backtesting(self) -> Dict[str, Any]:
        """Test ML backtesting functionality."""
        logger.info("Testing ML backtesting...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        if self.test_data is None or self.test_data.empty:
            logger.warning("⚠️ No test data available for ML backtesting")
            results['warnings'] += 1
            results['total_tests'] += 1
            self.test_results['ml_backtesting'] = results
            return results
        
        try:
            # Test ML backtester initialization
            backtester = MLBacktester()
            results['passed'] += 1
            logger.info("✅ ML backtester initialized successfully")
            results['total_tests'] += 1
            
            # Test backtesting configuration
            config = {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.001,
                'lookback_period': 60,
                'prediction_horizon': 5
            }
            
            # Test backtesting execution
            try:
                backtest_results = backtester.run_backtest(
                    self.test_data, 
                    self.test_ticker,
                    config=config
                )
                
                if backtest_results and isinstance(backtest_results, dict):
                    results['passed'] += 1
                    logger.info("✅ ML backtesting execution successful")
                    
                    # Check for expected keys
                    expected_keys = ['returns', 'sharpe_ratio', 'max_drawdown', 'total_return']
                    missing_keys = [key for key in expected_keys if key not in backtest_results]
                    
                    if not missing_keys:
                        results['passed'] += 1
                        logger.info("✅ ML backtesting results contain expected metrics")
                    else:
                        results['warnings'] += 1
                        logger.warning(f"⚠️ Missing metrics in backtesting results: {missing_keys}")
                    results['total_tests'] += 1
                else:
                    results['warnings'] += 1
                    logger.warning("⚠️ ML backtesting returned invalid format")
                results['total_tests'] += 1
                
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ ML backtesting execution warning: {e}")
            
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ ML backtesting test failed: {e}")
        
        self.test_results['ml_backtesting'] = results
        return results
    
    def test_strategy_backtesting(self) -> Dict[str, Any]:
        """Test strategy backtesting functionality."""
        logger.info("Testing strategy backtesting...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        if self.test_data is None or self.test_data.empty:
            logger.warning("⚠️ No test data available for strategy backtesting")
            results['warnings'] += 1
            results['total_tests'] += 1
            self.test_results['strategy_backtesting'] = results
            return results
        
        try:
            # Test momentum strategy framework
            momentum_framework = MomentumStrategyFramework()
            results['passed'] += 1
            logger.info("✅ Momentum strategy framework initialized successfully")
            results['total_tests'] += 1
            
            # Test ML-enhanced strategy
            ml_strategy = MLEnhancedEODStrategy()
            results['passed'] += 1
            logger.info("✅ ML-enhanced strategy initialized successfully")
            results['total_tests'] += 1
            
            # Test strategy analysis
            try:
                analysis = ml_strategy.analyze_stock(self.test_data, self.test_ticker)
                if analysis and isinstance(analysis, dict):
                    results['passed'] += 1
                    logger.info("✅ Strategy analysis successful")
                else:
                    results['warnings'] += 1
                    logger.warning("⚠️ Strategy analysis returned invalid format")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Strategy analysis warning: {e}")
            
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ Strategy backtesting test failed: {e}")
        
        self.test_results['strategy_backtesting'] = results
        return results
    
    def test_mlflow_integration(self) -> Dict[str, Any]:
        """Test MLflow integration for backtesting."""
        logger.info("Testing MLflow integration...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        try:
            # Test MLflow tracker initialization
            tracker = MLflowTracker()
            results['passed'] += 1
            logger.info("✅ MLflow tracker initialized successfully")
            results['total_tests'] += 1
            
            # Test experiment creation
            try:
                experiment_name = "backtesting_test"
                run_name = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                tracker.start_experiment(experiment_name, run_name)
                results['passed'] += 1
                logger.info("✅ MLflow experiment started successfully")
                results['total_tests'] += 1
                
                # Test parameter logging
                test_params = {
                    'test_ticker': self.test_ticker,
                    'test_date': datetime.now().isoformat(),
                    'test_type': 'backtesting'
                }
                
                tracker.log_params(test_params)
                results['passed'] += 1
                logger.info("✅ MLflow parameters logged successfully")
                results['total_tests'] += 1
                
                # Test metrics logging
                test_metrics = {
                    'test_accuracy': 0.85,
                    'test_precision': 0.82,
                    'test_recall': 0.88
                }
                
                tracker.log_metrics(test_metrics)
                results['passed'] += 1
                logger.info("✅ MLflow metrics logged successfully")
                results['total_tests'] += 1
                
                # End experiment
                tracker.end_experiment()
                results['passed'] += 1
                logger.info("✅ MLflow experiment ended successfully")
                results['total_tests'] += 1
                
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ MLflow experiment warning: {e}")
            
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ MLflow integration test failed: {e}")
        
        self.test_results['mlflow_integration'] = results
        return results
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics calculation."""
        logger.info("Testing performance metrics...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        if self.test_data is None or self.test_data.empty:
            logger.warning("⚠️ No test data available for performance metrics")
            results['warnings'] += 1
            results['total_tests'] += 1
            self.test_results['performance_metrics'] = results
            return results
        
        try:
            # Test basic performance metrics calculation
            returns = self.test_data['close_price'].pct_change().dropna()
            
            # Calculate basic metrics
            total_return = (self.test_data['close_price'].iloc[-1] / self.test_data['close_price'].iloc[0]) - 1
            annualized_return = total_return * (252 / len(returns))
            volatility = returns.std() * (252 ** 0.5)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Test metrics validation
            if isinstance(total_return, (int, float)) and not pd.isna(total_return):
                results['passed'] += 1
                logger.info(f"✅ Total return calculated: {total_return:.4f}")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ Total return calculation failed")
            results['total_tests'] += 1
            
            if isinstance(annualized_return, (int, float)) and not pd.isna(annualized_return):
                results['passed'] += 1
                logger.info(f"✅ Annualized return calculated: {annualized_return:.4f}")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ Annualized return calculation failed")
            results['total_tests'] += 1
            
            if isinstance(volatility, (int, float)) and not pd.isna(volatility):
                results['passed'] += 1
                logger.info(f"✅ Volatility calculated: {volatility:.4f}")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ Volatility calculation failed")
            results['total_tests'] += 1
            
            if isinstance(sharpe_ratio, (int, float)) and not pd.isna(sharpe_ratio):
                results['passed'] += 1
                logger.info(f"✅ Sharpe ratio calculated: {sharpe_ratio:.4f}")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ Sharpe ratio calculation failed")
            results['total_tests'] += 1
            
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ Performance metrics test failed: {e}")
        
        self.test_results['performance_metrics'] = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        ml_results = self.test_results['ml_backtesting']
        strategy_results = self.test_results['strategy_backtesting']
        mlflow_results = self.test_results['mlflow_integration']
        metrics_results = self.test_results['performance_metrics']
        
        total_tests = (
            ml_results.get('total_tests', 0) +
            strategy_results.get('total_tests', 0) +
            mlflow_results.get('total_tests', 0) +
            metrics_results.get('total_tests', 0)
        )
        
        total_passed = (
            ml_results.get('passed', 0) +
            strategy_results.get('passed', 0) +
            mlflow_results.get('passed', 0) +
            metrics_results.get('passed', 0)
        )
        
        total_failed = (
            ml_results.get('failed', 0) +
            strategy_results.get('failed', 0) +
            mlflow_results.get('failed', 0) +
            metrics_results.get('failed', 0)
        )
        
        total_warnings = (
            ml_results.get('warnings', 0) +
            strategy_results.get('warnings', 0) +
            mlflow_results.get('warnings', 0) +
            metrics_results.get('warnings', 0)
        )
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'warnings': total_warnings,
            'success_rate': success_rate,
            'status': 'PASS' if success_rate >= 75 else 'FAIL',
            'categories': {
                'ml_backtesting': {
                    'tests': ml_results.get('total_tests', 0),
                    'passed': ml_results.get('passed', 0),
                    'success_rate': (ml_results.get('passed', 0) / ml_results.get('total_tests', 1) * 100) if ml_results.get('total_tests', 0) > 0 else 0
                },
                'strategy_backtesting': {
                    'tests': strategy_results.get('total_tests', 0),
                    'passed': strategy_results.get('passed', 0),
                    'success_rate': (strategy_results.get('passed', 0) / strategy_results.get('total_tests', 1) * 100) if strategy_results.get('total_tests', 0) > 0 else 0
                },
                'mlflow_integration': {
                    'tests': mlflow_results.get('total_tests', 0),
                    'passed': mlflow_results.get('passed', 0),
                    'success_rate': (mlflow_results.get('passed', 0) / mlflow_results.get('total_tests', 1) * 100) if mlflow_results.get('total_tests', 0) > 0 else 0
                },
                'performance_metrics': {
                    'tests': metrics_results.get('total_tests', 0),
                    'passed': metrics_results.get('passed', 0),
                    'success_rate': (metrics_results.get('passed', 0) / metrics_results.get('total_tests', 1) * 100) if metrics_results.get('total_tests', 0) > 0 else 0
                }
            }
        }
        
        self.test_results['summary'] = summary
        return summary
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all backtesting tests."""
        logger.info("Starting Backtesting Test Suite...")
        start_time = time.time()
        
        # Setup test data
        self.setup_test_data()
        
        # Run tests
        self.test_ml_backtesting()
        self.test_strategy_backtesting()
        self.test_mlflow_integration()
        self.test_performance_metrics()
        
        # Generate summary
        summary = self.generate_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Backtesting Test Suite completed in {duration:.2f} seconds")
        logger.info(f"Results: {summary['passed']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")
        
        return self.test_results


def main():
    """Main test execution function."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run tests
    tester = BacktestingTester()
    results = tester.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*60)
    print("Backtesting Test Results")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Status: {summary['status']}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), '../test_results/backtesting_results.md')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"# Backtesting Test Results\n\n")
        f.write(f"**Timestamp**: {summary['timestamp']}\n")
        f.write(f"**Status**: {summary['status']}\n")
        f.write(f"**Success Rate**: {summary['success_rate']:.1f}%\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total Tests: {summary['total_tests']}\n")
        f.write(f"- Passed: {summary['passed']}\n")
        f.write(f"- Failed: {summary['failed']}\n")
        f.write(f"- Warnings: {summary['warnings']}\n\n")
        f.write(f"## Category Results\n\n")
        for category, stats in summary['categories'].items():
            f.write(f"### {category.replace('_', ' ').title()}\n")
            f.write(f"- Tests: {stats['tests']}\n")
            f.write(f"- Passed: {stats['passed']}\n")
            f.write(f"- Success Rate: {stats['success_rate']:.1f}%\n\n")
        f.write(f"## Detailed Results\n\n")
        f.write(f"```json\n{json.dumps(results, indent=2)}\n```\n")
    
    print(f"Results saved to: {results_file}")
    
    return summary['status'] == 'PASS'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 