#!/usr/bin/env python3
"""
Trading Strategies Test Suite
Tests all trading strategies including intraday, options, EOD, and ML strategies.
"""

import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..', 'src'))

from loguru import logger
from src.tools.enhanced_api import get_prices
from src.strategies.strategy_manager import get_strategy_summary
from src.strategies.intraday_strategies import (
    MomentumBreakoutStrategy, MarketDepthStrategy, VWAPStrategy,
    GapTradingStrategy, IntradayMeanReversionStrategy
)
from src.strategies.options_strategies import (
    IVSkewStrategy, GammaExposureStrategy, OptionsFlowStrategy,
    IronCondorStrategy, StraddleStrategy
)
from src.strategies.eod import (
    LongMomentumStrategy, ShortMomentumStrategy, MomentumStrategyFramework
)
from src.ml.ml_strategies import MLEnhancedEODStrategy


class StrategyTester:
    """Test all trading strategies."""
    
    def __init__(self):
        """Initialize the strategy tester."""
        self.test_results = {
            'intraday_strategies': {},
            'options_strategies': {},
            'eod_strategies': {},
            'ml_strategies': {},
            'summary': {},
            'errors': [],
            'warnings': []
        }
        
        # Test data
        self.test_ticker = 'AAPL'
        self.test_data = None
        
        logger.info("Strategy Tester initialized")
    
    def setup_test_data(self):
        """Setup test data for strategy testing."""
        try:
            self.test_data = get_prices(self.test_ticker, '2023-01-01', '2023-12-31')
            if self.test_data is not None and not self.test_data.empty:
                logger.info(f"✅ Test data loaded: {len(self.test_data)} records for {self.test_ticker}")
                return True
            else:
                logger.warning(f"⚠️ No test data available for {self.test_ticker}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            return False
    
    def test_intraday_strategies(self) -> Dict[str, Any]:
        """Test intraday trading strategies."""
        logger.info("Testing intraday strategies...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        strategies = {
            'Momentum Breakout': MomentumBreakoutStrategy(),
            'Market Depth': MarketDepthStrategy(),
            'VWAP': VWAPStrategy(),
            'Gap Trading': GapTradingStrategy(),
            'Intraday Mean Reversion': IntradayMeanReversionStrategy()
        }
        
        for strategy_name, strategy in strategies.items():
            logger.info(f"Testing {strategy_name} strategy")
            strategy_results = {
                'initialization': False,
                'analysis': False,
                'signal_generation': False
            }
            
            # Test initialization
            try:
                if strategy is not None:
                    strategy_results['initialization'] = True
                    results['passed'] += 1
                    logger.info(f"✅ {strategy_name} initialized successfully")
                else:
                    results['failed'] += 1
                    logger.error(f"❌ {strategy_name} initialization failed")
                results['total_tests'] += 1
            except Exception as e:
                results['failed'] += 1
                results['total_tests'] += 1
                logger.error(f"❌ {strategy_name} initialization error: {e}")
            
            # Test analysis (if data available)
            if self.test_data is not None and not self.test_data.empty:
                try:
                    analysis = strategy.analyze_stock(self.test_data, self.test_ticker)
                    if analysis and isinstance(analysis, dict):
                        strategy_results['analysis'] = True
                        results['passed'] += 1
                        logger.info(f"✅ {strategy_name} analysis successful")
                    else:
                        results['warnings'] += 1
                        logger.warning(f"⚠️ {strategy_name} analysis returned invalid format")
                    results['total_tests'] += 1
                except Exception as e:
                    results['warnings'] += 1
                    results['total_tests'] += 1
                    logger.warning(f"⚠️ {strategy_name} analysis warning: {e}")
            
            results['details'][strategy_name] = strategy_results
        
        self.test_results['intraday_strategies'] = results
        return results
    
    def test_options_strategies(self) -> Dict[str, Any]:
        """Test options trading strategies."""
        logger.info("Testing options strategies...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        strategies = {
            'IV Skew': IVSkewStrategy(),
            'Gamma Exposure': GammaExposureStrategy(),
            'Options Flow': OptionsFlowStrategy(),
            'Iron Condor': IronCondorStrategy(),
            'Straddle': StraddleStrategy()
        }
        
        for strategy_name, strategy in strategies.items():
            logger.info(f"Testing {strategy_name} strategy")
            strategy_results = {
                'initialization': False,
                'info_retrieval': False
            }
            
            # Test initialization
            try:
                if strategy is not None:
                    strategy_results['initialization'] = True
                    results['passed'] += 1
                    logger.info(f"✅ {strategy_name} initialized successfully")
                else:
                    results['failed'] += 1
                    logger.error(f"❌ {strategy_name} initialization failed")
                results['total_tests'] += 1
            except Exception as e:
                results['failed'] += 1
                results['total_tests'] += 1
                logger.error(f"❌ {strategy_name} initialization error: {e}")
            
            # Test strategy info retrieval
            try:
                info = strategy.get_strategy_info()
                if info and isinstance(info, dict):
                    strategy_results['info_retrieval'] = True
                    results['passed'] += 1
                    logger.info(f"✅ {strategy_name} info retrieved successfully")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ {strategy_name} info retrieval returned invalid format")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ {strategy_name} info retrieval warning: {e}")
            
            results['details'][strategy_name] = strategy_results
        
        self.test_results['options_strategies'] = results
        return results
    
    def test_eod_strategies(self) -> Dict[str, Any]:
        """Test EOD momentum strategies."""
        logger.info("Testing EOD strategies...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        strategies = {
            'Long Momentum': LongMomentumStrategy(),
            'Short Momentum': ShortMomentumStrategy(),
            'Momentum Framework': MomentumStrategyFramework()
        }
        
        for strategy_name, strategy in strategies.items():
            logger.info(f"Testing {strategy_name} strategy")
            strategy_results = {
                'initialization': False,
                'analysis': False
            }
            
            # Test initialization
            try:
                if strategy is not None:
                    strategy_results['initialization'] = True
                    results['passed'] += 1
                    logger.info(f"✅ {strategy_name} initialized successfully")
                else:
                    results['failed'] += 1
                    logger.error(f"❌ {strategy_name} initialization failed")
                results['total_tests'] += 1
            except Exception as e:
                results['failed'] += 1
                results['total_tests'] += 1
                logger.error(f"❌ {strategy_name} initialization error: {e}")
            
            # Test analysis (if data available and strategy supports it)
            if (self.test_data is not None and not self.test_data.empty and 
                hasattr(strategy, 'analyze_stock')):
                try:
                    analysis = strategy.analyze_stock(self.test_data, self.test_ticker)
                    if analysis and isinstance(analysis, dict):
                        strategy_results['analysis'] = True
                        results['passed'] += 1
                        logger.info(f"✅ {strategy_name} analysis successful")
                    else:
                        results['warnings'] += 1
                        logger.warning(f"⚠️ {strategy_name} analysis returned invalid format")
                    results['total_tests'] += 1
                except Exception as e:
                    results['warnings'] += 1
                    results['total_tests'] += 1
                    logger.warning(f"⚠️ {strategy_name} analysis warning: {e}")
            
            results['details'][strategy_name] = strategy_results
        
        self.test_results['eod_strategies'] = results
        return results
    
    def test_ml_strategies(self) -> Dict[str, Any]:
        """Test ML-enhanced strategies."""
        logger.info("Testing ML strategies...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        # Test ML-enhanced EOD strategy
        try:
            ml_strategy = MLEnhancedEODStrategy()
            results['passed'] += 1
            logger.info("✅ ML-enhanced EOD strategy initialized successfully")
            results['total_tests'] += 1
            
            # Test strategy summary
            summary = ml_strategy.get_strategy_summary()
            if summary and isinstance(summary, dict):
                results['passed'] += 1
                logger.info("✅ ML strategy summary retrieved successfully")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ ML strategy summary returned invalid format")
            results['total_tests'] += 1
            
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ ML strategy test failed: {e}")
        
        self.test_results['ml_strategies'] = results
        return results
    
    def test_strategy_manager(self) -> Dict[str, Any]:
        """Test strategy manager functionality."""
        logger.info("Testing strategy manager...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        try:
            summary = get_strategy_summary()
            if summary and isinstance(summary, dict):
                results['passed'] += 1
                logger.info("✅ Strategy manager summary retrieved successfully")
                
                # Check for expected keys
                expected_keys = ['total_strategies', 'strategy_categories']
                missing_keys = [key for key in expected_keys if key not in summary]
                
                if not missing_keys:
                    results['passed'] += 1
                    logger.info("✅ Strategy manager summary contains expected keys")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ Missing keys in strategy summary: {missing_keys}")
                results['total_tests'] += 1
            else:
                results['failed'] += 1
                logger.error("❌ Strategy manager summary returned invalid format")
            results['total_tests'] += 1
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ Strategy manager test failed: {e}")
        
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        intraday_results = self.test_results['intraday_strategies']
        options_results = self.test_results['options_strategies']
        eod_results = self.test_results['eod_strategies']
        ml_results = self.test_results['ml_strategies']
        
        total_tests = (
            intraday_results.get('total_tests', 0) +
            options_results.get('total_tests', 0) +
            eod_results.get('total_tests', 0) +
            ml_results.get('total_tests', 0)
        )
        
        total_passed = (
            intraday_results.get('passed', 0) +
            options_results.get('passed', 0) +
            eod_results.get('passed', 0) +
            ml_results.get('passed', 0)
        )
        
        total_failed = (
            intraday_results.get('failed', 0) +
            options_results.get('failed', 0) +
            eod_results.get('failed', 0) +
            ml_results.get('failed', 0)
        )
        
        total_warnings = (
            intraday_results.get('warnings', 0) +
            options_results.get('warnings', 0) +
            eod_results.get('warnings', 0) +
            ml_results.get('warnings', 0)
        )
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'warnings': total_warnings,
            'success_rate': success_rate,
            'status': 'PASS' if success_rate >= 80 else 'FAIL',
            'categories': {
                'intraday_strategies': {
                    'tests': intraday_results.get('total_tests', 0),
                    'passed': intraday_results.get('passed', 0),
                    'success_rate': (intraday_results.get('passed', 0) / intraday_results.get('total_tests', 1) * 100) if intraday_results.get('total_tests', 0) > 0 else 0
                },
                'options_strategies': {
                    'tests': options_results.get('total_tests', 0),
                    'passed': options_results.get('passed', 0),
                    'success_rate': (options_results.get('passed', 0) / options_results.get('total_tests', 1) * 100) if options_results.get('total_tests', 0) > 0 else 0
                },
                'eod_strategies': {
                    'tests': eod_results.get('total_tests', 0),
                    'passed': eod_results.get('passed', 0),
                    'success_rate': (eod_results.get('passed', 0) / eod_results.get('total_tests', 1) * 100) if eod_results.get('total_tests', 0) > 0 else 0
                },
                'ml_strategies': {
                    'tests': ml_results.get('total_tests', 0),
                    'passed': ml_results.get('passed', 0),
                    'success_rate': (ml_results.get('passed', 0) / ml_results.get('total_tests', 1) * 100) if ml_results.get('total_tests', 0) > 0 else 0
                }
            }
        }
        
        self.test_results['summary'] = summary
        return summary
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all strategy tests."""
        logger.info("Starting Strategy Test Suite...")
        start_time = time.time()
        
        # Setup test data
        self.setup_test_data()
        
        # Run tests
        self.test_intraday_strategies()
        self.test_options_strategies()
        self.test_eod_strategies()
        self.test_ml_strategies()
        
        # Generate summary
        summary = self.generate_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Strategy Test Suite completed in {duration:.2f} seconds")
        logger.info(f"Results: {summary['passed']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")
        
        return self.test_results


def main():
    """Main test execution function."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run tests
    tester = StrategyTester()
    results = tester.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*60)
    print("Strategy Test Results")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Status: {summary['status']}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), '../test_results/strategy_results.md')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"# Strategy Test Results\n\n")
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