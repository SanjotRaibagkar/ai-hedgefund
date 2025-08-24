#!/usr/bin/env python3
"""
Data Infrastructure Test Suite
Tests data retrieval and storage capabilities for Indian and US stocks.
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
from src.tools.enhanced_api import get_prices, get_financial_metrics, get_market_cap
from src.data.database.duckdb_manager import DatabaseManager
from src.data.collectors.async_data_collector import AsyncDataCollector
from src.data.update.update_manager import UpdateManager


class DataInfrastructureTester:
    """Test data infrastructure capabilities."""
    
    def __init__(self):
        """Initialize the data infrastructure tester."""
        self.test_results = {
            'data_retrieval': {},
            'data_storage': {},
            'data_quality': {},
            'update_mechanisms': {},
            'summary': {},
            'errors': [],
            'warnings': []
        }
        
        # Test tickers
        self.us_tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.indian_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        
        # Initialize database manager
        try:
            self.db_manager = DatabaseManager()
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            self.db_manager = None
        
        logger.info("Data Infrastructure Tester initialized")
    
    def test_data_retrieval(self) -> Dict[str, Any]:
        """Test data retrieval from various sources."""
        logger.info("Testing data retrieval...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        # Test US stock data retrieval
        for ticker in self.us_tickers:
            logger.info(f"Testing US stock data retrieval: {ticker}")
            ticker_results = {
                'prices': False,
                'financial_metrics': False,
                'market_cap': False
            }
            
            # Test price data
            try:
                prices = get_prices(ticker, '2023-01-01', '2023-12-31')
                if prices is not None and not prices.empty:
                    ticker_results['prices'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Price data retrieved for {ticker}: {len(prices)} records")
                else:
                    results['failed'] += 1
                    logger.warning(f"⚠️ No price data for {ticker}")
                results['total_tests'] += 1
            except Exception as e:
                results['failed'] += 1
                results['total_tests'] += 1
                logger.error(f"❌ Price data error for {ticker}: {e}")
            
            # Test financial metrics
            try:
                metrics = get_financial_metrics(ticker, '2023-12-31')
                if metrics is not None:
                    ticker_results['financial_metrics'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Financial metrics retrieved for {ticker}")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No financial metrics for {ticker}")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Financial metrics warning for {ticker}: {e}")
            
            # Test market cap
            try:
                market_cap = get_market_cap(ticker, '2023-12-31')
                if market_cap is not None:
                    ticker_results['market_cap'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Market cap retrieved for {ticker}: {market_cap}")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No market cap for {ticker}")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Market cap warning for {ticker}: {e}")
            
            results['details'][ticker] = ticker_results
        
        # Test Indian stock data retrieval
        for ticker in self.indian_tickers:
            logger.info(f"Testing Indian stock data retrieval: {ticker}")
            ticker_results = {
                'prices': False,
                'financial_metrics': False,
                'market_cap': False
            }
            
            # Test price data
            try:
                prices = get_prices(ticker, '2023-01-01', '2023-12-31')
                if prices is not None and not prices.empty:
                    ticker_results['prices'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Price data retrieved for {ticker}: {len(prices)} records")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No price data for {ticker} (may be expected)")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Price data warning for {ticker}: {e}")
            
            # Test financial metrics
            try:
                metrics = get_financial_metrics(ticker, '2023-12-31')
                if metrics is not None:
                    ticker_results['financial_metrics'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Financial metrics retrieved for {ticker}")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No financial metrics for {ticker}")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Financial metrics warning for {ticker}: {e}")
            
            # Test market cap
            try:
                market_cap = get_market_cap(ticker, '2023-12-31')
                if market_cap is not None:
                    ticker_results['market_cap'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Market cap retrieved for {ticker}: {market_cap}")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No market cap for {ticker}")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Market cap warning for {ticker}: {e}")
            
            results['details'][ticker] = ticker_results
        
        self.test_results['data_retrieval'] = results
        return results
    
    def test_data_storage(self) -> Dict[str, Any]:
        """Test data storage capabilities."""
        logger.info("Testing data storage...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        if self.db_manager is None:
            logger.error("Database manager not available, skipping storage tests")
            results['warnings'] += 1
            results['total_tests'] += 1
            self.test_results['data_storage'] = results
            return results
        
        # Test database initialization
        try:
            # Test table creation
            self.db_manager.create_tables()
            results['passed'] += 1
            logger.info("✅ Database tables created successfully")
            results['total_tests'] += 1
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ Database table creation failed: {e}")
        
        # Test data insertion
        test_ticker = 'AAPL'
        try:
            # Get test data
            prices = get_prices(test_ticker, '2023-01-01', '2023-01-31')
            if prices is not None and not prices.empty:
                # Test technical data insertion
                self.db_manager.insert_technical_data(test_ticker, prices)
                results['passed'] += 1
                logger.info(f"✅ Technical data inserted for {test_ticker}")
                results['total_tests'] += 1
                
                # Test data retrieval
                stored_data = self.db_manager.get_technical_data(test_ticker, '2023-01-01', '2023-01-31')
                if stored_data is not None and not stored_data.empty:
                    results['passed'] += 1
                    logger.info(f"✅ Technical data retrieved for {test_ticker}")
                else:
                    results['failed'] += 1
                    logger.error(f"❌ Technical data retrieval failed for {test_ticker}")
                results['total_tests'] += 1
            else:
                results['warnings'] += 1
                logger.warning(f"⚠️ No test data available for {test_ticker}")
                results['total_tests'] += 1
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ Data storage test failed: {e}")
        
        # Test data quality metrics
        try:
            quality_metrics = self.db_manager.get_data_quality_metrics()
            if quality_metrics is not None:
                results['passed'] += 1
                logger.info("✅ Data quality metrics retrieved")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ No data quality metrics available")
            results['total_tests'] += 1
        except Exception as e:
            results['warnings'] += 1
            results['total_tests'] += 1
            logger.warning(f"⚠️ Data quality metrics warning: {e}")
        
        self.test_results['data_storage'] = results
        return results
    
    def test_data_quality(self) -> Dict[str, Any]:
        """Test data quality validation."""
        logger.info("Testing data quality...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        # Test data format validation
        test_ticker = 'AAPL'
        try:
            prices = get_prices(test_ticker, '2023-01-01', '2023-01-31')
            if prices is not None and not prices.empty:
                # Check required columns
                required_columns = ['trade_date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
                missing_columns = [col for col in required_columns if col not in prices.columns]
                
                if not missing_columns:
                    results['passed'] += 1
                    logger.info("✅ Data format validation passed")
                else:
                    results['failed'] += 1
                    logger.error(f"❌ Missing columns: {missing_columns}")
                results['total_tests'] += 1
                
                # Check data types
                if prices['close_price'].dtype in ['float64', 'float32', 'int64', 'int32']:
                    results['passed'] += 1
                    logger.info("✅ Data types validation passed")
                else:
                    results['failed'] += 1
                    logger.error(f"❌ Invalid data types: {prices['close_price'].dtype}")
                results['total_tests'] += 1
                
                # Check for missing values
                missing_values = prices.isnull().sum().sum()
                if missing_values == 0:
                    results['passed'] += 1
                    logger.info("✅ No missing values found")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ Found {missing_values} missing values")
                results['total_tests'] += 1
                
                # Check data range
                if prices['close_price'].min() > 0:
                    results['passed'] += 1
                    logger.info("✅ Price data range validation passed")
                else:
                    results['failed'] += 1
                    logger.error("❌ Invalid price data (negative values)")
                results['total_tests'] += 1
            else:
                results['warnings'] += 1
                logger.warning(f"⚠️ No test data available for quality validation")
                results['total_tests'] += 1
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ Data quality test failed: {e}")
        
        self.test_results['data_quality'] = results
        return results
    
    def test_update_mechanisms(self) -> Dict[str, Any]:
        """Test data update mechanisms."""
        logger.info("Testing update mechanisms...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        # Test update manager initialization
        try:
            update_manager = UpdateManager()
            results['passed'] += 1
            logger.info("✅ Update manager initialized successfully")
            results['total_tests'] += 1
        except Exception as e:
            results['failed'] += 1
            results['total_tests'] += 1
            logger.error(f"❌ Update manager initialization failed: {e}")
        
        # Test data collector initialization
        try:
            if self.db_manager is not None:
                collector = AsyncDataCollector(self.db_manager)
                results['passed'] += 1
                logger.info("✅ Data collector initialized successfully")
            else:
                results['warnings'] += 1
                logger.warning("⚠️ Database manager not available for collector test")
            results['total_tests'] += 1
        except Exception as e:
            results['warnings'] += 1
            results['total_tests'] += 1
            logger.warning(f"⚠️ Data collector initialization warning: {e}")
        
        # Test missing data detection
        if self.db_manager is not None:
            try:
                missing_dates = self.db_manager.get_missing_data_dates('AAPL', '2023-01-01', '2023-01-31')
                if missing_dates is not None:
                    results['passed'] += 1
                    logger.info(f"✅ Missing data detection: {len(missing_dates)} missing dates")
                else:
                    results['warnings'] += 1
                    logger.warning("⚠️ Missing data detection returned None")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Missing data detection warning: {e}")
        
        self.test_results['update_mechanisms'] = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        retrieval_results = self.test_results['data_retrieval']
        storage_results = self.test_results['data_storage']
        quality_results = self.test_results['data_quality']
        update_results = self.test_results['update_mechanisms']
        
        total_tests = (
            retrieval_results.get('total_tests', 0) +
            storage_results.get('total_tests', 0) +
            quality_results.get('total_tests', 0) +
            update_results.get('total_tests', 0)
        )
        
        total_passed = (
            retrieval_results.get('passed', 0) +
            storage_results.get('passed', 0) +
            quality_results.get('passed', 0) +
            update_results.get('passed', 0)
        )
        
        total_failed = (
            retrieval_results.get('failed', 0) +
            storage_results.get('failed', 0) +
            quality_results.get('failed', 0) +
            update_results.get('failed', 0)
        )
        
        total_warnings = (
            retrieval_results.get('warnings', 0) +
            storage_results.get('warnings', 0) +
            quality_results.get('warnings', 0) +
            update_results.get('warnings', 0)
        )
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'warnings': total_warnings,
            'success_rate': success_rate,
            'status': 'PASS' if success_rate >= 70 else 'FAIL',
            'categories': {
                'data_retrieval': {
                    'tests': retrieval_results.get('total_tests', 0),
                    'passed': retrieval_results.get('passed', 0),
                    'success_rate': (retrieval_results.get('passed', 0) / retrieval_results.get('total_tests', 1) * 100) if retrieval_results.get('total_tests', 0) > 0 else 0
                },
                'data_storage': {
                    'tests': storage_results.get('total_tests', 0),
                    'passed': storage_results.get('passed', 0),
                    'success_rate': (storage_results.get('passed', 0) / storage_results.get('total_tests', 1) * 100) if storage_results.get('total_tests', 0) > 0 else 0
                },
                'data_quality': {
                    'tests': quality_results.get('total_tests', 0),
                    'passed': quality_results.get('passed', 0),
                    'success_rate': (quality_results.get('passed', 0) / quality_results.get('total_tests', 1) * 100) if quality_results.get('total_tests', 0) > 0 else 0
                },
                'update_mechanisms': {
                    'tests': update_results.get('total_tests', 0),
                    'passed': update_results.get('passed', 0),
                    'success_rate': (update_results.get('passed', 0) / update_results.get('total_tests', 1) * 100) if update_results.get('total_tests', 0) > 0 else 0
                }
            }
        }
        
        self.test_results['summary'] = summary
        return summary
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all data infrastructure tests."""
        logger.info("Starting Data Infrastructure Test Suite...")
        start_time = time.time()
        
        # Run tests
        self.test_data_retrieval()
        self.test_data_storage()
        self.test_data_quality()
        self.test_update_mechanisms()
        
        # Generate summary
        summary = self.generate_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Data Infrastructure Test Suite completed in {duration:.2f} seconds")
        logger.info(f"Results: {summary['passed']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")
        
        return self.test_results


def main():
    """Main test execution function."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run tests
    tester = DataInfrastructureTester()
    results = tester.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*60)
    print("Data Infrastructure Test Results")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Status: {summary['status']}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), '../test_results/data_infrastructure_results.md')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"# Data Infrastructure Test Results\n\n")
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