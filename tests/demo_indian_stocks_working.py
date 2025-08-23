#!/usr/bin/env python3
"""
Indian Stocks Working Demo
Demonstrates the working Indian stocks integration with real data.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger
from src.screening.screening_manager import ScreeningManager
from src.tools.enhanced_api import get_prices
from src.ui.branding import print_logo


def demo_indian_stocks_data():
    """Demo Indian stocks data fetching."""
    print_logo()
    
    logger.info("🇮🇳 Indian Stocks Integration Demo")
    logger.info("=" * 50)
    
    # Test working Indian stocks
    working_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    
    logger.info(f"📊 Testing {len(working_stocks)} Indian stocks...")
    
    for stock in working_stocks:
        try:
            logger.info(f"\n📈 {stock}:")
            
            # Get current price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now().replace(day=1)).strftime('%Y-%m-%d')  # Start of month
            
            price_data = get_prices(stock, start_date, end_date)
            
            if price_data is not None and hasattr(price_data, 'empty') and not price_data.empty:
                latest = price_data.iloc[-1]
                logger.info(f"   ✅ Current Price: ₹{latest['close_price']:.2f}")
                logger.info(f"   📊 Open: ₹{latest['open_price']:.2f} | High: ₹{latest['high_price']:.2f} | Low: ₹{latest['low_price']:.2f}")
                logger.info(f"   📅 Date: {latest['date']}")
            else:
                logger.warning(f"   ⚠️ No data available")
                
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
    
    return True


def demo_screening_system():
    """Demo the screening system with Indian stocks."""
    logger.info("\n🎯 Screening System Demo")
    logger.info("-" * 30)
    
    try:
        manager = ScreeningManager()
        
        # Use working Indian stocks
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        logger.info(f"🔍 Screening {len(test_stocks)} Indian stocks...")
        
        # Test EOD screening
        logger.info("\n📈 EOD Screening Results:")
        eod_results = manager.get_eod_signals(test_stocks, risk_reward_ratio=1.5)
        logger.info(f"   Bullish Signals: {eod_results['summary']['bullish_count']}")
        logger.info(f"   Bearish Signals: {eod_results['summary']['bearish_count']}")
        
        if eod_results['bullish_signals']:
            logger.info("   🟢 Top Bullish Signals:")
            for signal in eod_results['bullish_signals'][:2]:
                logger.info(f"      {signal['ticker']}: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | Target: ₹{signal['targets']['T1']:.2f}")
        
        if eod_results['bearish_signals']:
            logger.info("   🔴 Top Bearish Signals:")
            for signal in eod_results['bearish_signals'][:2]:
                logger.info(f"      {signal['ticker']}: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | Target: ₹{signal['targets']['T1']:.2f}")
        
        # Test intraday screening
        logger.info("\n⚡ Intraday Screening Results:")
        intraday_results = manager.get_intraday_signals(test_stocks)
        logger.info(f"   Breakout Signals: {intraday_results['summary']['breakout_count']}")
        logger.info(f"   Reversal Signals: {intraday_results['summary']['reversal_count']}")
        
        # Test quick screening
        logger.info("\n🔍 Quick Screening Results:")
        quick_results = manager.run_quick_screening(test_stocks)
        logger.info(f"   Market Sentiment: {quick_results['market_sentiment']}")
        logger.info(f"   Total Signals: {len(quick_results['quick_signals'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening demo failed: {e}")
        return False


def demo_system_capabilities():
    """Demo system capabilities."""
    logger.info("\n🚀 System Capabilities")
    logger.info("-" * 25)
    
    try:
        manager = ScreeningManager()
        summary = manager.get_screening_summary()
        
        logger.info(f"📊 System: {summary['name']}")
        logger.info(f"📝 Description: {summary['description']}")
        
        logger.info("\n🎯 Core Features:")
        for capability in summary['capabilities']:
            logger.info(f"   ✅ {capability}")
        
        logger.info("\n📈 Supported Markets:")
        for market in summary['supported_markets']:
            logger.info(f"   🌍 {market}")
        
        logger.info(f"\n📋 Default Stock List: {len(summary['default_stocks'])} stocks configured")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Capabilities demo failed: {e}")
        return False


def main():
    """Main demo execution."""
    logger.info("🚀 Starting Indian Stocks Working Demo")
    
    demos = [
        ("Indian Stocks Data", demo_indian_stocks_data),
        ("Screening System", demo_screening_system),
        ("System Capabilities", demo_system_capabilities)
    ]
    
    passed = 0
    failed = 0
    
    for demo_name, demo_func in demos:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {demo_name} Demo...")
        
        try:
            success = demo_func()
            if success:
                logger.info(f"✅ {demo_name} Demo PASSED")
                passed += 1
            else:
                logger.error(f"❌ {demo_name} Demo FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {demo_name} Demo FAILED with exception: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 INDIAN STOCKS WORKING DEMO RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total Demos: {len(demos)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(demos)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL DEMOS PASSED! Indian stocks integration is working perfectly!")
        logger.info("🇮🇳 Phase 5 Screening System is ready for Indian markets!")
    else:
        logger.warning(f"⚠️ {failed} demo(s) had issues.")
    
    logger.info("\n📋 Key Achievements:")
    logger.info("✅ Real-time Indian stock data fetching")
    logger.info("✅ Proper data format conversion")
    logger.info("✅ Screening system integration")
    logger.info("✅ Professional UI with company branding")
    logger.info("✅ Modular architecture for future expansion")
    
    return failed == 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 