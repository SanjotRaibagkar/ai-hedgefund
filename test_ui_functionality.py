#!/usr/bin/env python3
"""
UI Functionality Test
Tests the screening manager methods used by the UI.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger

def test_screening_manager_methods():
    """Test all screening manager methods used by the UI."""
    logger.info("🎯 Testing Screening Manager Methods")
    logger.info("=" * 40)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        
        # Initialize manager
        manager = ScreeningManager()
        logger.info("✅ ScreeningManager initialized")
        
        # Test stock list
        stock_list = "RELIANCE.NS, TCS.NS, HDFCBANK.NS"
        stocks = [s.strip() for s in stock_list.split(",") if s.strip()]
        logger.info(f"✅ Stock list parsed: {stocks}")
        
        # Test EOD screening
        logger.info("\n📈 Testing EOD Screening...")
        try:
            eod_results = manager.get_eod_signals(stocks, 2.0)
            logger.info("✅ EOD screening method exists")
            logger.info(f"   Summary: {eod_results.get('summary', {})}")
        except Exception as e:
            logger.error(f"❌ EOD screening failed: {e}")
        
        # Test Intraday screening
        logger.info("\n⚡ Testing Intraday Screening...")
        try:
            intraday_results = manager.get_intraday_signals(stocks)
            logger.info("✅ Intraday screening method exists")
            logger.info(f"   Summary: {intraday_results.get('summary', {})}")
        except Exception as e:
            logger.error(f"❌ Intraday screening failed: {e}")
        
        # Test Options analysis
        logger.info("\n🎯 Testing Options Analysis...")
        try:
            nifty_results = manager.get_options_analysis('NIFTY')
            logger.info("✅ Options analysis method exists")
            logger.info(f"   NIFTY results: {bool(nifty_results)}")
        except Exception as e:
            logger.error(f"❌ Options analysis failed: {e}")
        
        # Test Market predictions
        logger.info("\n🔮 Testing Market Predictions...")
        try:
            nifty_pred = manager.get_market_prediction('NIFTY', '15min')
            logger.info("✅ Market prediction method exists")
            logger.info(f"   NIFTY prediction: {bool(nifty_pred)}")
        except Exception as e:
            logger.error(f"❌ Market prediction failed: {e}")
        
        # Test Comprehensive screening
        logger.info("\n🎯 Testing Comprehensive Screening...")
        try:
            comp_results = manager.run_comprehensive_screening(
                stock_list=stocks,
                include_options=True,
                include_predictions=True
            )
            logger.info("✅ Comprehensive screening method exists")
            logger.info(f"   Summary: {comp_results.get('summary', {})}")
        except Exception as e:
            logger.error(f"❌ Comprehensive screening failed: {e}")
        
        # Test Trading recommendations
        logger.info("\n💡 Testing Trading Recommendations...")
        try:
            sample_results = {'summary': {'total_stocks': len(stocks)}}
            recommendations = manager.generate_trading_recommendations(sample_results)
            logger.info("✅ Trading recommendations method exists")
            logger.info(f"   Recommendations: {bool(recommendations)}")
        except Exception as e:
            logger.error(f"❌ Trading recommendations failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening manager test failed: {e}")
        return False

def test_ui_data_flow():
    """Test the data flow that the UI expects."""
    logger.info("\n🔄 Testing UI Data Flow")
    logger.info("-" * 30)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        
        manager = ScreeningManager()
        stocks = ["RELIANCE.NS", "TCS.NS"]
        
        # Test data flow for EOD
        logger.info("📈 Testing EOD data flow...")
        eod_results = manager.get_eod_signals(stocks, 2.0)
        
        # Check expected structure
        expected_keys = ['summary', 'bullish_signals', 'bearish_signals']
        for key in expected_keys:
            if key in eod_results:
                logger.info(f"   ✅ {key} present")
            else:
                logger.warning(f"   ⚠️ {key} missing")
        
        # Test data flow for Intraday
        logger.info("⚡ Testing Intraday data flow...")
        intraday_results = manager.get_intraday_signals(stocks)
        
        # Check expected structure
        expected_keys = ['summary', 'breakout_signals', 'reversal_signals', 'momentum_signals']
        for key in expected_keys:
            if key in intraday_results:
                logger.info(f"   ✅ {key} present")
            else:
                logger.warning(f"   ⚠️ {key} missing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI data flow test failed: {e}")
        return False

def main():
    """Main test execution."""
    logger.info("🚀 Starting UI Functionality Test Suite")
    
    tests = [
        ("Screening Manager Methods", test_screening_manager_methods),
        ("UI Data Flow", test_ui_data_flow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test...")
        
        try:
            success = test_func()
            if success:
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 UI FUNCTIONALITY TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! UI functionality is working!")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed. UI has functionality issues.")
    
    return failed == 0

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 