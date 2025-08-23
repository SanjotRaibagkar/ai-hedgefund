#!/usr/bin/env python3
"""
Simple Phase 5 Test Script
Tests the screening system with US stocks to verify functionality.
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.screening.screening_manager import ScreeningManager


def test_screening_with_us_stocks():
    """Test screening system with US stocks."""
    logger.info("Testing Screening System with US Stocks...")
    
    try:
        manager = ScreeningManager()
        
        # Use US stocks for testing
        test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        logger.info("Testing EOD Screener...")
        eod_results = manager.get_eod_signals(test_stocks, risk_reward_ratio=1.5)
        
        logger.info(f"EOD Results: {eod_results['summary']['bullish_count']} bullish, {eod_results['summary']['bearish_count']} bearish signals")
        
        logger.info("Testing Intraday Screener...")
        intraday_results = manager.get_intraday_signals(test_stocks)
        
        logger.info(f"Intraday Results: {intraday_results['summary']['breakout_count']} breakouts, {intraday_results['summary']['reversal_count']} reversals")
        
        logger.info("Testing Quick Screening...")
        quick_results = manager.run_quick_screening(test_stocks)
        
        logger.info(f"Quick Screening: {len(quick_results['quick_signals'])} signals, {quick_results['market_sentiment']} sentiment")
        
        logger.info("✅ All screening tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening test failed: {e}")
        return False


def test_screening_manager():
    """Test screening manager functionality."""
    logger.info("Testing Screening Manager...")
    
    try:
        manager = ScreeningManager()
        
        # Test summary
        summary = manager.get_screening_summary()
        logger.info(f"Screening System: {summary['name']}")
        logger.info(f"Description: {summary['description']}")
        logger.info(f"Capabilities: {len(summary['capabilities'])}")
        
        # Test default stocks
        default_stocks = manager.default_indian_stocks
        logger.info(f"Default stocks: {len(default_stocks)} stocks configured")
        
        logger.info("✅ Screening Manager test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening Manager test failed: {e}")
        return False


def main():
    """Main test execution."""
    logger.info("Starting Simple Phase 5 Tests...")
    
    tests = [
        ("Screening Manager", test_screening_manager),
        ("Screening with US Stocks", test_screening_with_us_stocks)
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
            
            if success:
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
    logger.info("SIMPLE PHASE 5 TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Phase 5 Screening System is working.")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed.")
    
    return failed == 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 