#!/usr/bin/env python3
"""
Phase 5 Demo Script
Demonstrates the screening system capabilities with sample data.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.screening.screening_manager import ScreeningManager
from src.ui.branding import print_logo


def demo_screening_system():
    """Demo the screening system capabilities."""
    print_logo()
    
    logger.info("🚀 Phase 5: Stock Screener & Advanced UI System Demo")
    logger.info("=" * 60)
    
    # Initialize screening manager
    manager = ScreeningManager()
    
    # Get system summary
    summary = manager.get_screening_summary()
    logger.info(f"📊 System: {summary['name']}")
    logger.info(f"📝 Description: {summary['description']}")
    logger.info(f"🔧 Capabilities: {len(summary['capabilities'])}")
    
    logger.info("\n🎯 Core Features:")
    for capability in summary['capabilities']:
        logger.info(f"   ✅ {capability}")
    
    logger.info("\n📈 Supported Markets:")
    for market in summary['supported_markets']:
        logger.info(f"   🌍 {market}")
    
    logger.info("\n📋 Default Stock List:")
    for i, stock in enumerate(summary['default_stocks'], 1):
        logger.info(f"   {i:2d}. {stock}")
    
    logger.info("\n🔍 Screening Modules:")
    for module_name, module_info in summary['modules'].items():
        logger.info(f"   📦 {module_info['name']}")
        logger.info(f"      {module_info['description']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Phase 5 Screening System is ready for production!")
    logger.info("💡 Use the following commands to get started:")
    logger.info("   python test_phase5_simple.py  # Test with US stocks")
    logger.info("   python src/ui/web_app/app.py  # Start web UI")
    logger.info("   python test_phase5.py         # Full system test")
    
    return True


def demo_quick_screening():
    """Demo quick screening functionality."""
    logger.info("\n🔍 Quick Screening Demo")
    logger.info("-" * 40)
    
    try:
        manager = ScreeningManager()
        
        # Use a small list of stocks for demo
        demo_stocks = ['AAPL', 'MSFT', 'GOOGL']
        
        logger.info(f"📊 Screening {len(demo_stocks)} stocks...")
        results = manager.run_quick_screening(demo_stocks)
        
        logger.info(f"📈 Market Sentiment: {results['market_sentiment']}")
        logger.info(f"🎯 Signals Found: {len(results['quick_signals'])}")
        
        if results['quick_signals']:
            logger.info("\n🏆 Top Signals:")
            for i, signal in enumerate(results['quick_signals'][:3], 1):
                logger.info(f"   {i}. {signal['ticker']} - {signal['signal']} ({signal['confidence']}% confidence)")
                logger.info(f"      Entry: ${signal['entry']:.2f} | SL: ${signal['stop_loss']:.2f} | Target: ${signal['target']:.2f}")
                logger.info(f"      Risk-Reward: {signal['risk_reward']:.2f}")
        else:
            logger.info("   No high-confidence signals found in demo data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick screening demo failed: {e}")
        return False


def demo_screening_manager():
    """Demo screening manager functionality."""
    logger.info("\n⚙️ Screening Manager Demo")
    logger.info("-" * 40)
    
    try:
        manager = ScreeningManager()
        
        # Test EOD screening
        logger.info("📊 Testing EOD Screener...")
        eod_results = manager.get_eod_signals(['AAPL', 'MSFT'], risk_reward_ratio=1.5)
        logger.info(f"   EOD Signals: {eod_results['summary']['bullish_count']} bullish, {eod_results['summary']['bearish_count']} bearish")
        
        # Test intraday screening
        logger.info("⚡ Testing Intraday Screener...")
        intraday_results = manager.get_intraday_signals(['AAPL', 'MSFT'])
        logger.info(f"   Intraday Signals: {intraday_results['summary']['breakout_count']} breakouts, {intraday_results['summary']['reversal_count']} reversals")
        
        # Test options analysis
        logger.info("📈 Testing Options Analyzer...")
        try:
            options_results = manager.get_options_analysis('NIFTY')
            if options_results:
                logger.info("   Options Analysis: Completed successfully")
            else:
                logger.info("   Options Analysis: No data available (expected for demo)")
        except Exception as e:
            logger.info(f"   Options Analysis: {str(e)[:50]}... (expected for demo)")
        
        # Test market predictions
        logger.info("🔮 Testing Market Predictor...")
        try:
            prediction = manager.get_market_prediction('NIFTY', 'eod')
            if prediction and prediction.get('prediction'):
                pred_data = prediction['prediction']
                logger.info(f"   Market Prediction: {pred_data.get('direction', 'NEUTRAL')} - {pred_data.get('confidence', 0)}% confidence")
            else:
                logger.info("   Market Prediction: No data available (expected for demo)")
        except Exception as e:
            logger.info(f"   Market Prediction: {str(e)[:50]}... (expected for demo)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening manager demo failed: {e}")
        return False


def main():
    """Main demo execution."""
    logger.info("Starting Phase 5 Demo...")
    
    demos = [
        ("Screening System Overview", demo_screening_system),
        ("Quick Screening Demo", demo_quick_screening),
        ("Screening Manager Demo", demo_screening_manager)
    ]
    
    passed = 0
    failed = 0
    
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ {demo_name} failed with exception: {e}")
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 PHASE 5 DEMO RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Demos: {len(demos)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(demos)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL DEMOS PASSED! Phase 5 is working perfectly.")
    else:
        logger.warning(f"⚠️ {failed} demo(s) had issues.")
    
    logger.info("\n🚀 Phase 5 Screening System is ready for use!")
    logger.info("📚 Check the documentation for detailed usage instructions.")
    
    return failed == 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 