#!/usr/bin/env python3
"""
Phase 5 Test Script
Tests the comprehensive screening system.
"""

import sys
import os
import time
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.screening.screening_manager import ScreeningManager


def test_eod_screener():
    """Test EOD stock screener."""
    logger.info("Testing EOD Stock Screener...")
    
    try:
        manager = ScreeningManager()
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        results = manager.get_eod_signals(test_stocks, risk_reward_ratio=1.5)
        
        assert 'bullish_signals' in results
        assert 'bearish_signals' in results
        assert 'summary' in results
        
        summary = results['summary']
        logger.info(f"✅ EOD Screener: {summary['bullish_count']} bullish, {summary['bearish_count']} bearish signals")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ EOD Screener test failed: {e}")
        return False


def test_intraday_screener():
    """Test intraday stock screener."""
    logger.info("Testing Intraday Stock Screener...")
    
    try:
        manager = ScreeningManager()
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        results = manager.get_intraday_signals(test_stocks)
        
        assert 'breakout_signals' in results
        assert 'reversal_signals' in results
        assert 'momentum_signals' in results
        
        summary = results['summary']
        logger.info(f"✅ Intraday Screener: {summary['breakout_count']} breakouts, {summary['reversal_count']} reversals")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Intraday Screener test failed: {e}")
        return False


def test_options_analyzer():
    """Test options analyzer."""
    logger.info("Testing Options Analyzer...")
    
    try:
        manager = ScreeningManager()
        
        results = manager.get_options_analysis('NIFTY')
        
        assert 'index' in results
        assert 'current_price' in results
        assert 'analysis' in results
        
        analysis = results['analysis']
        assert 'oi_analysis' in analysis
        assert 'volatility_analysis' in analysis
        
        logger.info(f"✅ Options Analyzer: Nifty at {results['current_price']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Options Analyzer test failed: {e}")
        return False


def test_market_predictor():
    """Test market predictor."""
    logger.info("Testing Market Predictor...")
    
    try:
        manager = ScreeningManager()
        
        prediction = manager.get_market_prediction('NIFTY', 'eod')
        
        assert 'index' in prediction
        assert 'timeframe' in prediction
        assert 'prediction' in prediction
        
        pred_data = prediction['prediction']
        if pred_data:
            assert 'direction' in pred_data
            assert 'confidence' in pred_data
            
            logger.info(f"✅ Market Predictor: {pred_data['direction']} - {pred_data['confidence']}% confidence")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Market Predictor test failed: {e}")
        return False


def test_comprehensive_screening():
    """Test comprehensive screening."""
    logger.info("Testing Comprehensive Screening...")
    
    try:
        manager = ScreeningManager()
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        results = manager.run_comprehensive_screening(
            stock_list=test_stocks,
            include_options=True,
            include_predictions=True
        )
        
        assert 'stock_screening' in results
        assert 'options_analysis' in results
        assert 'market_predictions' in results
        assert 'summary' in results
        
        summary = results['summary']
        logger.info(f"✅ Comprehensive Screening: {summary['eod_signals']} EOD, {summary['intraday_signals']} intraday signals")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive Screening test failed: {e}")
        return False


def test_quick_screening():
    """Test quick screening."""
    logger.info("Testing Quick Screening...")
    
    try:
        manager = ScreeningManager()
        
        results = manager.run_quick_screening()
        
        assert 'quick_signals' in results
        assert 'market_sentiment' in results
        
        logger.info(f"✅ Quick Screening: {len(results['quick_signals'])} signals, {results['market_sentiment']} sentiment")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick Screening test failed: {e}")
        return False


def main():
    """Main test execution."""
    logger.info("Starting Phase 5 Screening System Tests...")
    
    tests = [
        ("EOD Screener", test_eod_screener),
        ("Intraday Screener", test_intraday_screener),
        ("Options Analyzer", test_options_analyzer),
        ("Market Predictor", test_market_predictor),
        ("Comprehensive Screening", test_comprehensive_screening),
        ("Quick Screening", test_quick_screening)
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
    logger.info("PHASE 5 TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Phase 5 Screening System is ready.")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed.")
    
    return failed == 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 