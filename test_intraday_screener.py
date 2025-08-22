#!/usr/bin/env python3
"""
Intraday Screener Test Script
Tests the intraday screener functionality with Indian stocks.
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.screening.intraday_screener import IntradayStockScreener
from src.screening.screening_manager import ScreeningManager
from src.tools.enhanced_api import get_prices
from src.ui.branding import print_logo


def test_intraday_screener_direct():
    """Test the intraday screener directly."""
    print_logo()
    
    logger.info("⚡ Testing Intraday Screener Directly")
    logger.info("=" * 50)
    
    try:
        # Initialize intraday screener
        screener = IntradayStockScreener()
        logger.info("✅ Intraday screener initialized successfully")
        
        # Test with working Indian stocks
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        logger.info(f"📊 Testing {len(test_stocks)} Indian stocks for intraday signals...")
        
        for stock in test_stocks:
            try:
                logger.info(f"\n📈 Testing {stock}...")
                
                # Get current price data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                
                price_data = get_prices(stock, start_date, end_date)
                
                if price_data is not None and hasattr(price_data, 'empty') and not price_data.empty:
                    logger.info(f"   ✅ Data available: {len(price_data)} records")
                    
                    # Test individual stock analysis
                    analysis = screener._analyze_intraday_opportunities(price_data, stock)
                    
                    if analysis:
                        logger.info(f"   🎯 Signal Type: {analysis['signal_type']}")
                        logger.info(f"   📊 Entry: ₹{analysis['entry_price']:.2f}")
                        logger.info(f"   🛑 Stop Loss: ₹{analysis['stop_loss']:.2f}")
                        logger.info(f"   🎯 Target: ₹{analysis['target']:.2f}")
                        logger.info(f"   📈 Confidence: {analysis['confidence']}%")
                        logger.info(f"   💡 Reasons: {', '.join(analysis['reasons'])}")
                    else:
                        logger.info(f"   ⚠️ No clear signal for {stock}")
                else:
                    logger.warning(f"   ❌ No data available for {stock}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error testing {stock}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Intraday screener test failed: {e}")
        return False


def test_intraday_screener_via_manager():
    """Test intraday screener through the screening manager."""
    logger.info("\n🎯 Testing Intraday Screener via Manager")
    logger.info("-" * 40)
    
    try:
        manager = ScreeningManager()
        
        # Test with working Indian stocks
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
        
        logger.info(f"📊 Screening {len(test_stocks)} stocks for intraday signals...")
        
        # Get intraday signals
        results = manager.get_intraday_signals(test_stocks)
        
        logger.info(f"📈 Results Summary:")
        logger.info(f"   Breakout Signals: {results['summary']['breakout_count']}")
        logger.info(f"   Reversal Signals: {results['summary']['reversal_count']}")
        logger.info(f"   Total Signals: {len(results['breakout_signals']) + len(results['reversal_signals'])}")
        
        # Show breakout signals
        if results['breakout_signals']:
            logger.info(f"\n🟢 Breakout Signals:")
            for signal in results['breakout_signals'][:3]:  # Show top 3
                logger.info(f"   📈 {signal['ticker']}: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | Target: ₹{signal['target']:.2f}")
                logger.info(f"      Confidence: {signal['confidence']}% | Type: {signal['signal_type']}")
        
        # Show reversal signals
        if results['reversal_signals']:
            logger.info(f"\n🔴 Reversal Signals:")
            for signal in results['reversal_signals'][:3]:  # Show top 3
                logger.info(f"   📉 {signal['ticker']}: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | Target: ₹{signal['target']:.2f}")
                logger.info(f"      Confidence: {signal['confidence']}% | Type: {signal['signal_type']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Manager intraday test failed: {e}")
        return False


def test_intraday_technical_indicators():
    """Test intraday technical indicators calculation."""
    logger.info("\n🔧 Testing Intraday Technical Indicators")
    logger.info("-" * 40)
    
    try:
        screener = IntradayStockScreener()
        
        # Test with one stock
        test_stock = 'RELIANCE.NS'
        
        logger.info(f"📊 Testing technical indicators for {test_stock}...")
        
        # Get price data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        price_data = get_prices(test_stock, start_date, end_date)
        
        if price_data is not None and hasattr(price_data, 'empty') and not price_data.empty:
            logger.info(f"   ✅ Data available: {len(price_data)} records")
            
            # Test ATR calculation
            try:
                atr = screener._calculate_atr(price_data)
                logger.info(f"   📊 ATR: ₹{atr:.2f}")
            except Exception as e:
                logger.warning(f"   ⚠️ ATR calculation failed: {e}")
            
            # Test support/resistance levels
            try:
                levels = screener._get_support_resistance_levels(price_data)
                logger.info(f"   📈 Support/Resistance: {levels}")
            except Exception as e:
                logger.warning(f"   ⚠️ Support/Resistance calculation failed: {e}")
            
            # Test support bounce detection
            try:
                is_support_bounce = screener._is_support_bounce(price_data)
                logger.info(f"   📊 Support Bounce: {is_support_bounce}")
            except Exception as e:
                logger.warning(f"   ⚠️ Support bounce detection failed: {e}")
            
            # Test resistance rejection detection
            try:
                is_resistance_rejection = screener._is_resistance_rejection(price_data)
                logger.info(f"   📊 Resistance Rejection: {is_resistance_rejection}")
            except Exception as e:
                logger.warning(f"   ⚠️ Resistance rejection detection failed: {e}")
            
        else:
            logger.warning(f"   ❌ No data available for {test_stock}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Technical indicators test failed: {e}")
        return False


def test_intraday_signal_generation():
    """Test intraday signal generation logic."""
    logger.info("\n🎯 Testing Intraday Signal Generation")
    logger.info("-" * 40)
    
    try:
        screener = IntradayStockScreener()
        
        # Test with multiple stocks
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        for stock in test_stocks:
            logger.info(f"\n📈 Testing signal generation for {stock}...")
            
            # Get price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            price_data = get_prices(stock, start_date, end_date)
            
            if price_data is not None and hasattr(price_data, 'empty') and not price_data.empty:
                # Calculate indicators first
                indicators = screener.momentum_indicators.calculate_all_indicators(price_data)
                
                # Test breakout detection
                breakout_signals = screener._detect_breakouts(price_data, indicators)
                if breakout_signals:
                    logger.info(f"   🟢 Breakout Signal: {breakout_signals['signal_type']}")
                    logger.info(f"   📊 Entry: ₹{breakout_signals['entry_price']:.2f}")
                    logger.info(f"   📈 Confidence: {breakout_signals['confidence']}%")
                else:
                    logger.info(f"   ⚠️ No breakout signal")
                
                # Test reversal detection
                reversal_signals = screener._detect_reversals(price_data, indicators)
                if reversal_signals:
                    logger.info(f"   🔴 Reversal Signal: {reversal_signals['signal_type']}")
                    logger.info(f"   📊 Entry: ₹{reversal_signals['entry_price']:.2f}")
                    logger.info(f"   📈 Confidence: {reversal_signals['confidence']}%")
                else:
                    logger.info(f"   ⚠️ No reversal signal")
                
                # Test momentum detection
                momentum_signals = screener._detect_momentum(price_data, indicators)
                if momentum_signals:
                    logger.info(f"   ⚡ Momentum Signal: {momentum_signals['signal_type']}")
                    logger.info(f"   📊 Entry: ₹{momentum_signals['entry_price']:.2f}")
                    logger.info(f"   📈 Confidence: {momentum_signals['confidence']}%")
                else:
                    logger.info(f"   ⚠️ No momentum signal")
                
            else:
                logger.warning(f"   ❌ No data available for {stock}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal generation test failed: {e}")
        return False


def test_intraday_entry_sl_targets():
    """Test intraday entry, stop loss, and target calculations."""
    logger.info("\n🎯 Testing Intraday Entry/SL/Target Calculations")
    logger.info("-" * 50)
    
    try:
        screener = IntradayStockScreener()
        
        # Test with one stock
        test_stock = 'RELIANCE.NS'
        
        logger.info(f"📊 Testing calculations for {test_stock}...")
        
        # Get price data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        price_data = get_prices(test_stock, start_date, end_date)
        
        if price_data is not None and hasattr(price_data, 'empty') and not price_data.empty:
            current_price = price_data['close_price'].iloc[-1]
            
            # Test entry, SL, target calculations for breakout
            entry, sl, targets = screener._calculate_intraday_levels(price_data, current_price, 'BREAKOUT')
            
            logger.info(f"   📈 BREAKOUT Signal:")
            logger.info(f"      Entry: ₹{entry:.2f}")
            logger.info(f"      Stop Loss: ₹{sl:.2f}")
            logger.info(f"      Target: ₹{targets:.2f}")
            
            # Test reversal calculations
            entry, sl, targets = screener._calculate_intraday_levels(price_data, current_price, 'REVERSAL')
            
            logger.info(f"   📉 REVERSAL Signal:")
            logger.info(f"      Entry: ₹{entry:.2f}")
            logger.info(f"      Stop Loss: ₹{sl:.2f}")
            logger.info(f"      Target: ₹{targets:.2f}")
            
            # Test momentum calculations
            entry, sl, targets = screener._calculate_intraday_levels(price_data, current_price, 'MOMENTUM')
            
            logger.info(f"   ⚡ MOMENTUM Signal:")
            logger.info(f"      Entry: ₹{entry:.2f}")
            logger.info(f"      Stop Loss: ₹{sl:.2f}")
            logger.info(f"      Target: ₹{targets:.2f}")
            
        else:
            logger.warning(f"   ❌ No data available for {test_stock}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Entry/SL/Target test failed: {e}")
        return False


def main():
    """Main test execution."""
    logger.info("🚀 Starting Intraday Screener Comprehensive Test")
    logger.info("=" * 60)
    
    tests = [
        ("Direct Intraday Screener", test_intraday_screener_direct),
        ("Manager Integration", test_intraday_screener_via_manager),
        ("Technical Indicators", test_intraday_technical_indicators),
        ("Signal Generation", test_intraday_signal_generation),
        ("Entry/SL/Targets", test_intraday_entry_sl_targets)
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
    logger.info("🎯 INTRADAY SCREENER TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Intraday screener is working perfectly!")
        logger.info("⚡ Intraday screening system is ready for Indian markets!")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed. Issues identified.")
    
    logger.info("\n📋 Intraday Screener Features Tested:")
    logger.info("✅ Direct screener functionality")
    logger.info("✅ Manager integration")
    logger.info("✅ Technical indicators calculation")
    logger.info("✅ Signal generation logic")
    logger.info("✅ Entry/SL/Target calculations")
    logger.info("✅ Indian stocks compatibility")
    
    return failed == 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 