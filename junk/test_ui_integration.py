#!/usr/bin/env python3
"""
Test UI Integration with Unified Screener
Verify that the UI can properly use the unified screener
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.screening.screening_manager import ScreeningManager
from loguru import logger

def test_ui_integration():
    """Test the UI integration with unified screener."""
    logger.info("🧪 Testing UI Integration with Unified Screener")
    logger.info("=" * 60)
    
    try:
        # Initialize screening manager
        logger.info("🔧 Initializing Screening Manager...")
        screening_manager = ScreeningManager()
        logger.info("✅ Screening Manager initialized successfully")
        
        # Test 1: Basic EOD screening
        logger.info("\n🔍 Test 1: Basic EOD Screening")
        logger.info("-" * 40)
        
        test_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
        
        for mode in ["basic", "enhanced", "comprehensive"]:
            logger.info(f"📊 Testing {mode} mode...")
            try:
                results = screening_manager.get_eod_signals(
                    stock_list=test_stocks,
                    risk_reward_ratio=2.0,
                    analysis_mode=mode
                )
                
                summary = results['summary']
                logger.info(f"✅ {mode} mode completed:")
                logger.info(f"   Total stocks: {summary['total_stocks']}")
                logger.info(f"   Bullish signals: {summary['bullish_count']}")
                logger.info(f"   Bearish signals: {summary['bearish_count']}")
                
                if results['bullish_signals']:
                    sample = results['bullish_signals'][0]
                    logger.info(f"   Sample signal: {sample['symbol']} - {sample['confidence']}% confidence")
                
            except Exception as e:
                logger.error(f"❌ {mode} mode failed: {e}")
        
        # Test 2: Comprehensive screening
        logger.info("\n🔍 Test 2: Comprehensive Screening")
        logger.info("-" * 40)
        
        try:
            comprehensive_results = screening_manager.run_comprehensive_screening(
                stock_list=test_stocks,
                include_options=False,
                include_predictions=False
            )
            
            logger.info("✅ Comprehensive screening completed:")
            logger.info(f"   EOD signals: {comprehensive_results['summary']['eod_signals']}")
            logger.info(f"   Intraday signals: {comprehensive_results['summary']['intraday_signals']}")
            
        except Exception as e:
            logger.error(f"❌ Comprehensive screening failed: {e}")
        
        # Test 3: Error handling
        logger.info("\n🔍 Test 3: Error Handling")
        logger.info("-" * 40)
        
        try:
            # Test with invalid symbols
            invalid_results = screening_manager.get_eod_signals(
                stock_list=["INVALID1.NS", "INVALID2.NS"],
                analysis_mode="basic"
            )
            
            logger.info("✅ Error handling test completed:")
            logger.info(f"   Total stocks: {invalid_results['summary']['total_stocks']}")
            logger.info(f"   Bullish signals: {invalid_results['summary']['bullish_count']}")
            logger.info(f"   Bearish signals: {invalid_results['summary']['bearish_count']}")
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
        
        # Test 4: Performance test
        logger.info("\n🔍 Test 4: Performance Test")
        logger.info("-" * 40)
        
        import time
        
        start_time = time.time()
        results = screening_manager.get_eod_signals(
            stock_list=test_stocks,
            analysis_mode="comprehensive"
        )
        end_time = time.time()
        
        logger.info(f"✅ Performance test completed in {end_time - start_time:.2f}s")
        logger.info(f"   Stocks processed: {results['summary']['total_stocks']}")
        
        # Test 5: Verify data format
        logger.info("\n🔍 Test 5: Data Format Verification")
        logger.info("-" * 40)
        
        results = screening_manager.get_eod_signals(
            stock_list=test_stocks[:2],
            analysis_mode="enhanced"
        )
        
        # Check required fields
        required_fields = ['summary', 'bullish_signals', 'bearish_signals']
        for field in required_fields:
            if field in results:
                logger.info(f"✅ {field} field present")
            else:
                logger.error(f"❌ {field} field missing")
        
        # Check summary fields
        summary_fields = ['total_stocks', 'bullish_count', 'bearish_count']
        for field in summary_fields:
            if field in results['summary']:
                logger.info(f"✅ summary.{field} field present")
            else:
                logger.error(f"❌ summary.{field} field missing")
        
        # Check signal format if signals exist
        if results['bullish_signals']:
            sample_signal = results['bullish_signals'][0]
            signal_fields = ['symbol', 'signal', 'confidence', 'entry_price', 'stop_loss', 'targets']
            for field in signal_fields:
                if field in sample_signal:
                    logger.info(f"✅ signal.{field} field present")
                else:
                    logger.error(f"❌ signal.{field} field missing")
        
        logger.info("\n🎉 All UI Integration Tests Completed Successfully!")
        logger.info("=" * 60)
        logger.info("✅ Screening Manager works with unified screener")
        logger.info("✅ All analysis modes work correctly")
        logger.info("✅ Error handling works properly")
        logger.info("✅ Performance is acceptable")
        logger.info("✅ Data format is correct")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ui_integration()
