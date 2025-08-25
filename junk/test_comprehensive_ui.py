#!/usr/bin/env python3
"""
Test Comprehensive UI Screening
Verify that comprehensive screening works with all stocks from database
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
from datetime import datetime
from loguru import logger

from src.screening.screening_manager import ScreeningManager
from src.data.database.duckdb_manager import DatabaseManager

async def test_comprehensive_screening():
    """Test comprehensive screening with all stocks from database."""
    logger.info("🧪 Testing Comprehensive Screening with All Stocks")
    logger.info("=" * 60)
    
    try:
        # Initialize managers
        screening_manager = ScreeningManager()
        db_manager = DatabaseManager()
        
        # Get all available symbols
        all_symbols = db_manager.get_available_symbols()
        logger.info(f"📊 Total symbols in database: {len(all_symbols)}")
        
        # Test comprehensive screening
        logger.info("🚀 Running comprehensive screening...")
        start_time = datetime.now()
        
        results = screening_manager.run_comprehensive_screening(
            stock_list=None,  # Use all stocks from database
            include_options=True,
            include_predictions=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        logger.info(f"✅ Comprehensive screening completed in {duration:.2f}s")
        
        summary = results['summary']
        logger.info(f"📊 Results Summary:")
        logger.info(f"   Total Stocks Analyzed: {summary['total_stocks']}")
        logger.info(f"   📈 EOD Signals: {summary['eod_signals']}")
        logger.info(f"   ⚡ Intraday Signals: {summary['intraday_signals']}")
        logger.info(f"   🎯 Options Analysis: {summary['options_analysis_count']}")
        logger.info(f"   🔮 Predictions: {summary['predictions_count']}")
        
        # Show EOD signals
        eod_results = results.get('stock_screening', {}).get('eod', {})
        if eod_results.get('bullish_signals'):
            logger.info(f"\n📈 Bullish Signals Found: {len(eod_results['bullish_signals'])}")
            for i, signal in enumerate(eod_results['bullish_signals'][:5], 1):
                logger.info(f"   {i}. {signal.get('symbol', signal.get('ticker', 'Unknown'))} - {signal['confidence']}% confidence")
        
        if eod_results.get('bearish_signals'):
            logger.info(f"\n📉 Bearish Signals Found: {len(eod_results['bearish_signals'])}")
            for i, signal in enumerate(eod_results['bearish_signals'][:5], 1):
                logger.info(f"   {i}. {signal.get('symbol', signal.get('ticker', 'Unknown'))} - {signal['confidence']}% confidence")
        
        if not eod_results.get('bullish_signals') and not eod_results.get('bearish_signals'):
            logger.info("\n💡 No EOD signals found - this is normal with current criteria")
        
        # Test recommendations generation
        logger.info("\n🎯 Testing Trading Recommendations...")
        recommendations = screening_manager.generate_trading_recommendations(results)
        
        if recommendations.get('high_confidence_signals'):
            logger.info(f"High Confidence Signals: {len(recommendations['high_confidence_signals'])}")
        else:
            logger.info("No high confidence signals found")
        
        logger.info("✅ Comprehensive screening test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in comprehensive screening test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_comprehensive_screening())
