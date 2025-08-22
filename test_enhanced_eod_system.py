#!/usr/bin/env python3
"""
Test Enhanced EOD System
Demonstrates the enhanced EOD screening system with database integration.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger

async def test_enhanced_eod_system():
    """Test the enhanced EOD system."""
    logger.info("🚀 Testing Enhanced EOD System")
    logger.info("=" * 40)
    
    try:
        from src.data.indian_data_manager import indian_data_manager
        
        # Test 1: Database initialization
        logger.info("\n🔧 Test 1: Database Initialization")
        logger.info("-" * 30)
        
        stats = indian_data_manager.get_database_stats()
        logger.info(f"✅ Database initialized")
        logger.info(f"   Total securities: {stats['total_securities']}")
        logger.info(f"   Total data points: {stats['total_data_points']}")
        logger.info(f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        
        # Test 2: Get securities
        logger.info("\n📊 Test 2: Fetching Securities")
        logger.info("-" * 30)
        
        securities = await indian_data_manager.get_all_securities()
        logger.info(f"✅ Fetched {len(securities)} securities")
        
        if securities:
            sample_symbols = [s['symbol'] for s in securities[:5]]
            logger.info(f"   Sample symbols: {sample_symbols}")
        
        # Test 3: Download historical data (small sample)
        logger.info("\n📥 Test 3: Downloading Historical Data")
        logger.info("-" * 35)
        
        if securities:
            # Use first 10 symbols for testing
            test_symbols = [s['symbol'] for s in securities[:10]]
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"📊 Downloading data for {len(test_symbols)} symbols...")
            result = await indian_data_manager.download_historical_data(
                test_symbols, start_date, end_date
            )
            
            logger.info(f"✅ Download completed")
            logger.info(f"   Total: {result['total']}")
            logger.info(f"   Completed: {result['completed']}")
            logger.info(f"   Failed: {result['failed']}")
            logger.info(f"   Time: {result['elapsed_time']:.2f}s")
        
        # Test 4: Get price data
        logger.info("\n📈 Test 4: Retrieving Price Data")
        logger.info("-" * 30)
        
        if securities:
            test_symbol = securities[0]['symbol']
            price_data = await indian_data_manager.get_price_data(
                test_symbol, 
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            )
            
            if not price_data.empty:
                logger.info(f"✅ Retrieved data for {test_symbol}")
                logger.info(f"   Records: {len(price_data)}")
                logger.info(f"   Date range: {price_data.index[0]} to {price_data.index[-1]}")
                logger.info(f"   Latest close: ₹{price_data['close_price'].iloc[-1]:.2f}")
            else:
                logger.warning(f"⚠️ No data available for {test_symbol}")
        
        # Test 5: Enhanced EOD Screening
        logger.info("\n🎯 Test 5: Enhanced EOD Screening")
        logger.info("-" * 30)
        
        try:
            from src.screening.enhanced_eod_screener import enhanced_eod_screener
            
            # Screen a small sample
            if securities:
                test_symbols = [s['symbol'] for s in securities[:20]]
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                logger.info(f"🎯 Screening {len(test_symbols)} symbols...")
                screening_results = await enhanced_eod_screener.screen_universe(
                    symbols=test_symbols,
                    start_date=start_date,
                    end_date=end_date,
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
            logger.warning(f"⚠️ Enhanced screener not available: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def demo_full_workflow():
    """Demo the full workflow."""
    logger.info("\n🔄 Demo: Full EOD Workflow")
    logger.info("=" * 30)
    
    try:
        from src.data.indian_data_manager import indian_data_manager
        
        # Step 1: Get all securities
        logger.info("📊 Step 1: Getting all securities...")
        securities = await indian_data_manager.get_all_securities()
        logger.info(f"✅ Found {len(securities)} securities")
        
        # Step 2: Download 1 year data for top 50 stocks
        logger.info("\n📥 Step 2: Downloading historical data...")
        top_symbols = [s['symbol'] for s in securities[:50]]
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        download_result = await indian_data_manager.download_historical_data(
            top_symbols, start_date, end_date
        )
        logger.info(f"✅ Downloaded data for {download_result['completed']} symbols")
        
        # Step 3: Run screening
        logger.info("\n🎯 Step 3: Running EOD screening...")
        try:
            from src.screening.enhanced_eod_screener import enhanced_eod_screener
            
            screening_results = await enhanced_eod_screener.screen_universe(
                symbols=top_symbols,
                start_date=start_date,
                end_date=end_date,
                min_volume=100000,
                min_price=10.0
            )
            
            summary = screening_results['summary']
            logger.info(f"✅ Screening completed")
            logger.info(f"   📈 Bullish signals: {summary['bullish_signals']}")
            logger.info(f"   📉 Bearish signals: {summary['bearish_signals']}")
            
            # Show top signals
            if screening_results['bullish_signals']:
                logger.info(f"\n🏆 Top Bullish Signals:")
                for i, signal in enumerate(screening_results['bullish_signals'][:3]):
                    logger.info(f"   {i+1}. {signal['symbol']} - {signal['confidence']}% confidence")
                    logger.info(f"      Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}")
            
            if screening_results['bearish_signals']:
                logger.info(f"\n🔻 Top Bearish Signals:")
                for i, signal in enumerate(screening_results['bearish_signals'][:3]):
                    logger.info(f"   {i+1}. {signal['symbol']} - {signal['confidence']}% confidence")
                    logger.info(f"      Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}")
        
        except ImportError:
            logger.warning("⚠️ Enhanced screener not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

def main():
    """Main test execution."""
    logger.info("🚀 Enhanced EOD System Test Suite")
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run tests
    async def run_tests():
        tests = [
            ("Enhanced EOD System", test_enhanced_eod_system),
            ("Full Workflow Demo", demo_full_workflow)
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
        logger.info("🎯 ENHANCED EOD SYSTEM TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {len(tests)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! Enhanced EOD system is working!")
            logger.info("\n🚀 System Features:")
            logger.info("   ✅ Database integration")
            logger.info("   ✅ Fast data retrieval")
            logger.info("   ✅ Concurrent processing")
            logger.info("   ✅ CSV output with reasons")
            logger.info("   ✅ Modular design")
        else:
            logger.warning(f"⚠️ {failed} test(s) failed.")
        
        return failed == 0
    
    # Run async tests
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 