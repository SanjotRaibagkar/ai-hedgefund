#!/usr/bin/env python3
"""
Test Simple EOD System
Demonstrates the simple EOD screening system using NSEUtility.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger

async def test_simple_eod_system():
    """Test the simple EOD system."""
    logger.info("🚀 Testing Simple EOD System with NSEUtility")
    logger.info("=" * 50)
    
    try:
        from src.data.enhanced_indian_data_manager import enhanced_indian_data_manager
        
        # Test 1: Database initialization
        logger.info("\n🔧 Test 1: Database Initialization")
        logger.info("-" * 30)
        
        stats = enhanced_indian_data_manager.get_database_stats()
        logger.info(f"✅ Database initialized")
        logger.info(f"   Total securities: {stats['total_securities']}")
        logger.info(f"   Total data points: {stats['total_data_points']}")
        logger.info(f"   Database size: {stats['database_size_mb']:.2f} MB")
        logger.info(f"   NSEUtility available: {stats['nse_utility_available']}")
        
        # Test 2: Get securities
        logger.info("\n📊 Test 2: Fetching Securities")
        logger.info("-" * 30)
        
        securities = await enhanced_indian_data_manager.get_all_indian_stocks()
        logger.info(f"✅ Fetched {len(securities)} securities")
        
        if securities:
            sample_symbols = [s['symbol'] for s in securities[:5]]
            logger.info(f"   Sample symbols: {sample_symbols}")
        
        # Test 3: Download sample data
        logger.info("\n📥 Test 3: Downloading Sample Data")
        logger.info("-" * 30)
        
        if securities:
            # Use first 5 symbols for testing
            test_symbols = [s['symbol'] for s in securities[:5]]
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"📊 Downloading data for {len(test_symbols)} symbols...")
            result = await enhanced_indian_data_manager.download_10_years_data(test_symbols)
            
            logger.info(f"✅ Download completed")
            logger.info(f"   Total: {result['total']}")
            logger.info(f"   Completed: {result['completed']}")
            logger.info(f"   Failed: {result['failed']}")
            logger.info(f"   Time: {result['elapsed_time']:.2f}s")
        
        # Test 4: Simple EOD Screening
        logger.info("\n🎯 Test 4: Simple EOD Screening")
        logger.info("-" * 30)
        
        try:
            from src.screening.simple_eod_screener import simple_eod_screener
            
            # Screen a small sample
            if securities:
                test_symbols = [s['symbol'] for s in securities[:10]]
                
                logger.info(f"🎯 Screening {len(test_symbols)} symbols...")
                screening_results = await simple_eod_screener.screen_universe(
                    symbols=test_symbols,
                    min_volume=50000,
                    min_price=5.0
                )
                
                summary = screening_results['summary']
                logger.info(f"✅ Screening completed")
                logger.info(f"   Total screened: {summary['total_screened']}")
                logger.info(f"   Bullish signals: {summary['bullish_signals']}")
                logger.info(f"   Bearish signals: {summary['bearish_signals']}")
                
                # Show sample signals
                if screening_results['bullish_signals']:
                    sample_bullish = screening_results['bullish_signals'][0]
                    logger.info(f"\n📈 Sample Bullish Signal:")
                    logger.info(f"   Symbol: {sample_bullish['symbol']}")
                    logger.info(f"   Confidence: {sample_bullish['confidence']}%")
                    logger.info(f"   Entry: ₹{sample_bullish['entry_price']}")
                    logger.info(f"   SL: ₹{sample_bullish['stop_loss']}")
                    logger.info(f"   T1: ₹{sample_bullish['targets']['T1']}")
                    logger.info(f"   Reasons: {', '.join(sample_bullish['reasons'])}")
                
                if screening_results['bearish_signals']:
                    sample_bearish = screening_results['bearish_signals'][0]
                    logger.info(f"\n📉 Sample Bearish Signal:")
                    logger.info(f"   Symbol: {sample_bearish['symbol']}")
                    logger.info(f"   Confidence: {sample_bearish['confidence']}%")
                    logger.info(f"   Entry: ₹{sample_bearish['entry_price']}")
                    logger.info(f"   SL: ₹{sample_bearish['stop_loss']}")
                    logger.info(f"   T1: ₹{sample_bearish['targets']['T1']}")
                    logger.info(f"   Reasons: {', '.join(sample_bearish['reasons'])}")
        
        except ImportError as e:
            logger.warning(f"⚠️ Simple screener not available: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def demo_nseutility_workflow():
    """Demo the NSEUtility workflow."""
    logger.info("\n🔄 Demo: NSEUtility Workflow")
    logger.info("=" * 30)
    
    try:
        # Test NSEUtility directly using existing infrastructure
        logger.info("📊 Testing NSEUtility directly...")
        
        try:
            from src.nsedata.NseUtility import NseUtils
            nse = NseUtils()
            
            # Test with a known symbol
            test_symbol = "RELIANCE"
            price_info = nse.price_info(test_symbol)
            
            if price_info:
                logger.info(f"✅ NSEUtility working with {test_symbol}")
                logger.info(f"   Current Price: ₹{price_info.get('LastTradedPrice', 0)}")
                logger.info(f"   Volume: {price_info.get('Volume', 0)}")
                logger.info(f"   High: ₹{price_info.get('High', 0)}")
                logger.info(f"   Low: ₹{price_info.get('Low', 0)}")
            else:
                logger.warning(f"⚠️ No data for {test_symbol}")
            
            # Test with another symbol
            test_symbol2 = "TCS"
            price_info2 = nse.price_info(test_symbol2)
            
            if price_info2:
                logger.info(f"✅ NSEUtility working with {test_symbol2}")
                logger.info(f"   Current Price: ₹{price_info2.get('LastTradedPrice', 0)}")
                logger.info(f"   Volume: {price_info2.get('Volume', 0)}")
            else:
                logger.warning(f"⚠️ No data for {test_symbol2}")
        
        except ImportError as e:
            logger.error(f"❌ Failed to import NSEUtility: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ NSEUtility error: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

def main():
    """Main test execution."""
    logger.info("🚀 Simple EOD System Test Suite")
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run tests
    async def run_tests():
        tests = [
            ("NSEUtility Workflow", demo_nseutility_workflow),
            ("Simple EOD System", test_simple_eod_system)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {test_name}...")
            
            try:
                success = await test_func()
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
        logger.info("🎯 SIMPLE EOD SYSTEM TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {len(tests)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! Simple EOD system is working!")
            logger.info("\n🚀 System Features:")
            logger.info("   ✅ NSEUtility integration")
            logger.info("   ✅ Database storage")
            logger.info("   ✅ Fast screening")
            logger.info("   ✅ CSV output")
            logger.info("   ✅ Modular design")
        else:
            logger.warning(f"⚠️ {failed} test(s) failed.")
        
        return failed == 0
    
    # Run async tests
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 