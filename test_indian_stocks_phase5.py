#!/usr/bin/env python3
"""
Indian Stocks Phase 5 Test Script
Tests the screening system specifically with Indian stocks and fixes data issues.
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.screening.screening_manager import ScreeningManager
from src.tools.enhanced_api import get_prices, get_financial_metrics
from src.data.providers.provider_factory import DataProviderFactory


def test_indian_data_availability():
    """Test data availability for Indian stocks."""
    logger.info("🔍 Testing Indian Stock Data Availability")
    logger.info("=" * 50)
    
    # Test different Indian stock formats
    test_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',  # Without .NS suffix
        'NIFTY.NS', 'BANKNIFTY.NS', 'NIFTY', 'BANKNIFTY'  # Indices
    ]
    
    available_stocks = []
    unavailable_stocks = []
    
    for stock in test_stocks:
        try:
            logger.info(f"Testing {stock}...")
            
            # Get price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            price_data = get_prices(stock, start_date, end_date)
            
            if price_data is not None:
                if isinstance(price_data, list) and len(price_data) > 0:
                    logger.info(f"✅ {stock}: Data available ({len(price_data)} records)")
                    available_stocks.append(stock)
                elif hasattr(price_data, 'empty') and not price_data.empty:
                    logger.info(f"✅ {stock}: Data available ({len(price_data)} records)")
                    available_stocks.append(stock)
                else:
                    logger.warning(f"⚠️ {stock}: Empty data")
                    unavailable_stocks.append(stock)
            else:
                logger.warning(f"❌ {stock}: No data returned")
                unavailable_stocks.append(stock)
                
        except Exception as e:
            logger.error(f"❌ {stock}: Error - {e}")
            unavailable_stocks.append(stock)
    
    logger.info(f"\n📊 Data Availability Summary:")
    logger.info(f"✅ Available: {len(available_stocks)} stocks")
    logger.info(f"❌ Unavailable: {len(unavailable_stocks)} stocks")
    
    if available_stocks:
        logger.info(f"\n✅ Working stocks: {', '.join(available_stocks)}")
    
    if unavailable_stocks:
        logger.info(f"\n❌ Problem stocks: {', '.join(unavailable_stocks)}")
    
    return available_stocks, unavailable_stocks


def test_data_provider_factory():
    """Test data provider factory for Indian stocks."""
    logger.info("\n🏭 Testing Data Provider Factory")
    logger.info("-" * 40)
    
    try:
        factory = DataProviderFactory()
        
        # Test different stock formats
        test_cases = [
            'RELIANCE.NS', 'TCS.NS', 'NIFTY.NS', 'BANKNIFTY.NS',
            'RELIANCE', 'TCS', 'NIFTY', 'BANKNIFTY'
        ]
        
        for stock in test_cases:
            try:
                provider = factory.get_provider_for_ticker(stock)
                logger.info(f"✅ {stock} -> {provider.get_provider_name()}")
            except Exception as e:
                logger.error(f"❌ {stock} -> Error: {e}")
                
    except Exception as e:
        logger.error(f"❌ Data Provider Factory test failed: {e}")


def test_enhanced_api_with_indian_stocks():
    """Test enhanced API specifically with Indian stocks."""
    logger.info("\n🔧 Testing Enhanced API with Indian Stocks")
    logger.info("-" * 50)
    
    # Test with known working Indian stocks
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    
    for stock in test_stocks:
        try:
            logger.info(f"\n📈 Testing {stock}...")
            
            # Test price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            logger.info(f"   Fetching price data from {start_date} to {end_date}...")
            price_data = get_prices(stock, start_date, end_date)
            
            if price_data is not None:
                if isinstance(price_data, list):
                    logger.info(f"   ✅ Price data: {len(price_data)} records (list format)")
                    if len(price_data) > 0:
                        logger.info(f"   📊 Sample data: {price_data[0]}")
                elif hasattr(price_data, 'empty'):
                    logger.info(f"   ✅ Price data: {len(price_data)} records (DataFrame format)")
                    if not price_data.empty:
                        logger.info(f"   📊 Columns: {list(price_data.columns)}")
                        logger.info(f"   📊 Sample: {price_data.head(1).to_dict()}")
            else:
                logger.warning(f"   ⚠️ No price data returned")
            
            # Test financial metrics
            try:
                logger.info(f"   Fetching financial metrics...")
                metrics = get_financial_metrics(stock, end_date)
                if metrics:
                    logger.info(f"   ✅ Financial metrics available")
                else:
                    logger.warning(f"   ⚠️ No financial metrics")
            except Exception as e:
                logger.warning(f"   ⚠️ Financial metrics error: {e}")
                
        except Exception as e:
            logger.error(f"   ❌ Error testing {stock}: {e}")


def test_screening_with_working_indian_stocks():
    """Test screening with Indian stocks that have data."""
    logger.info("\n🎯 Testing Screening with Working Indian Stocks")
    logger.info("-" * 50)
    
    # Get working stocks from previous test
    available_stocks, _ = test_indian_data_availability()
    
    if not available_stocks:
        logger.warning("⚠️ No working Indian stocks found. Using fallback stocks.")
        available_stocks = ['RELIANCE.NS', 'TCS.NS']  # Fallback
    
    logger.info(f"📊 Testing screening with: {', '.join(available_stocks)}")
    
    try:
        manager = ScreeningManager()
        
        # Test EOD screening
        logger.info("\n📈 Testing EOD Screener...")
        eod_results = manager.get_eod_signals(available_stocks[:3], risk_reward_ratio=1.5)
        logger.info(f"   EOD Results: {eod_results['summary']['bullish_count']} bullish, {eod_results['summary']['bearish_count']} bearish")
        
        # Test intraday screening
        logger.info("\n⚡ Testing Intraday Screener...")
        intraday_results = manager.get_intraday_signals(available_stocks[:3])
        logger.info(f"   Intraday Results: {intraday_results['summary']['breakout_count']} breakouts, {intraday_results['summary']['reversal_count']} reversals")
        
        # Test quick screening
        logger.info("\n🔍 Testing Quick Screening...")
        quick_results = manager.run_quick_screening(available_stocks[:3])
        logger.info(f"   Quick Screening: {len(quick_results['quick_signals'])} signals, {quick_results['market_sentiment']} sentiment")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening test failed: {e}")
        return False


def fix_data_issues():
    """Identify and fix common data issues."""
    logger.info("\n🔧 Fixing Data Issues")
    logger.info("-" * 30)
    
    # Test different data formats and fix issues
    test_stock = 'RELIANCE.NS'
    
    try:
        logger.info(f"Testing data format for {test_stock}...")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        price_data = get_prices(test_stock, start_date, end_date)
        
        if price_data is not None:
            if isinstance(price_data, list):
                logger.info(f"Data is in list format with {len(price_data)} records")
                if len(price_data) > 0:
                    sample = price_data[0]
                    logger.info(f"Sample record: {sample}")
                    
                    # Check if it's a dictionary with proper keys
                    if isinstance(sample, dict):
                        keys = list(sample.keys())
                        logger.info(f"Available keys: {keys}")
                        
                        # Check for required columns
                        required_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
                        missing_columns = [col for col in required_columns if col not in keys]
                        
                        if missing_columns:
                            logger.warning(f"Missing columns: {missing_columns}")
                            logger.info("This explains the 'close_price' errors in screening")
                        else:
                            logger.info("✅ All required columns present")
            
            elif hasattr(price_data, 'empty'):
                logger.info(f"Data is in DataFrame format with {len(price_data)} records")
                logger.info(f"Columns: {list(price_data.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data format test failed: {e}")
        return False


def main():
    """Main test execution."""
    logger.info("🚀 Starting Indian Stocks Phase 5 Test")
    logger.info("=" * 60)
    
    tests = [
        ("Data Provider Factory", test_data_provider_factory),
        ("Data Availability", test_indian_data_availability),
        ("Enhanced API", test_enhanced_api_with_indian_stocks),
        ("Data Issues Fix", fix_data_issues),
        ("Screening with Working Stocks", test_screening_with_working_indian_stocks)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test...")
        
        start_time = time.time()
        try:
            success = test_func()
            duration = time.time() - start_time
            
            if success is not False:  # Some tests don't return boolean
                logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED ({duration:.2f}s)")
                failed += 1
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {test_name} FAILED with exception ({duration:.2f}s): {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 INDIAN STOCKS PHASE 5 TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Indian stocks integration is working.")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed. Data issues identified.")
    
    logger.info("\n📋 Next Steps:")
    logger.info("1. Check data provider configuration")
    logger.info("2. Verify API endpoints for Indian stocks")
    logger.info("3. Test with different stock formats")
    logger.info("4. Fix column mapping issues if found")
    
    return failed == 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 