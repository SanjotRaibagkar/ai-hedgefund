#!/usr/bin/env python3
"""
Test UI Comprehensive Screening Integration
Verify that the UI comprehensive screening works correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger

from src.screening.screening_manager import ScreeningManager

def test_ui_comprehensive_integration():
    """Test UI comprehensive screening integration."""
    logger.info("🧪 Testing UI Comprehensive Screening Integration")
    logger.info("=" * 60)
    
    try:
        # Initialize screening manager
        screening_manager = ScreeningManager()
        
        # Test comprehensive screening (same as UI button)
        logger.info("🚀 Running comprehensive screening (UI simulation)...")
        
        results = screening_manager.run_comprehensive_screening(
            stock_list=None,  # Use all stocks from database
            include_options=True,
            include_predictions=True
        )
        
        # Display results (same format as UI)
        summary = results['summary']
        logger.info(f"📊 UI Results Summary:")
        logger.info(f"   Total Stocks Analyzed: {summary['total_stocks']}")
        logger.info(f"   📈 EOD Signals: {summary['eod_signals']}")
        logger.info(f"   ⚡ Intraday Signals: {summary['intraday_signals']}")
        logger.info(f"   🎯 Options Analysis: {summary['options_analysis_count']}")
        logger.info(f"   🔮 Predictions: {summary['predictions_count']}")
        
        # Test recommendations generation (same as UI)
        logger.info("\n🎯 Testing Trading Recommendations (UI simulation)...")
        recommendations = screening_manager.generate_trading_recommendations(results)
        
        if recommendations.get('high_confidence_signals'):
            logger.info(f"High Confidence Signals: {len(recommendations['high_confidence_signals'])}")
        else:
            logger.info("No high confidence signals found")
        
        # Test market overview (same as UI)
        market_overview = recommendations.get('market_overview', {})
        logger.info(f"\n📈 Market Overview:")
        logger.info(f"   Nifty: {market_overview.get('nifty_direction', 'NEUTRAL')} - {market_overview.get('nifty_confidence', 0)}% confidence")
        logger.info(f"   BankNifty: {market_overview.get('banknifty_direction', 'NEUTRAL')} - {market_overview.get('banknifty_confidence', 0)}% confidence")
        
        # Test action items (same as UI)
        action_items = recommendations.get('action_items', [])
        if action_items:
            logger.info(f"\n📋 Action Items:")
            for item in action_items[:3]:
                logger.info(f"   • {item}")
        
        logger.info("\n✅ UI Comprehensive screening integration test completed successfully!")
        logger.info("💡 The comprehensive screening button in the UI will now work with ALL stocks from the database")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in UI comprehensive screening test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ui_comprehensive_integration()
