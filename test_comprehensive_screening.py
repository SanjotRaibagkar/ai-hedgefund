#!/usr/bin/env python3
"""
Test Comprehensive Screening Functionality
Check what's working, what's not, and where dummy data is used
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
import json
from datetime import datetime

def test_comprehensive_screening():
    """Test comprehensive screening functionality."""
    logger.info("🧪 Testing Comprehensive Screening Functionality")
    logger.info("=" * 60)
    
    try:
        # Import screening manager
        from src.screening.screening_manager import ScreeningManager
        
        # Initialize screening manager
        logger.info("📊 Initializing Screening Manager...")
        screening_manager = ScreeningManager()
        
        # Test comprehensive screening
        logger.info("🚀 Running Comprehensive Screening...")
        start_time = datetime.now()
        
        results = screening_manager.run_comprehensive_screening(
            stock_list=None,  # Use all stocks from database
            include_options=True,
            include_predictions=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"⏱️ Screening completed in {duration:.2f} seconds")
        
        # Analyze results
        logger.info("\n📋 RESULTS ANALYSIS:")
        logger.info("=" * 40)
        
        # Check summary
        summary = results.get('summary', {})
        logger.info(f"📊 Summary Statistics:")
        logger.info(f"   Total Stocks: {summary.get('total_stocks', 'N/A')}")
        logger.info(f"   EOD Signals: {summary.get('eod_signals', 'N/A')}")
        logger.info(f"   Intraday Signals: {summary.get('intraday_signals', 'N/A')}")
        logger.info(f"   Options Analysis: {summary.get('options_analysis_count', 'N/A')}")
        logger.info(f"   Predictions: {summary.get('predictions_count', 'N/A')}")
        
        # Check for errors
        if 'error' in results:
            logger.error(f"❌ Error in screening: {results['error']}")
        
        # Check EOD screening results
        eod_results = results.get('stock_screening', {}).get('eod', {})
        if eod_results:
            logger.info(f"\n📈 EOD Screening Results:")
            logger.info(f"   Bullish Signals: {len(eod_results.get('bullish_signals', []))}")
            logger.info(f"   Bearish Signals: {len(eod_results.get('bearish_signals', []))}")
            
            # Check sample signals for dummy data
            if eod_results.get('bullish_signals'):
                sample_signal = eod_results['bullish_signals'][0]
                logger.info(f"   Sample Bullish Signal: {sample_signal.get('symbol', 'N/A')} - Confidence: {sample_signal.get('confidence', 'N/A')}%")
            
            if eod_results.get('bearish_signals'):
                sample_signal = eod_results['bearish_signals'][0]
                logger.info(f"   Sample Bearish Signal: {sample_signal.get('symbol', 'N/A')} - Confidence: {sample_signal.get('confidence', 'N/A')}%")
        else:
            logger.warning("⚠️ No EOD screening results found")
        
        # Check intraday screening results
        intraday_results = results.get('stock_screening', {}).get('intraday', {})
        if intraday_results:
            logger.info(f"\n⚡ Intraday Screening Results:")
            logger.info(f"   Breakout Signals: {len(intraday_results.get('breakout_signals', []))}")
            logger.info(f"   Reversal Signals: {len(intraday_results.get('reversal_signals', []))}")
            logger.info(f"   Momentum Signals: {len(intraday_results.get('momentum_signals', []))}")
        else:
            logger.warning("⚠️ No intraday screening results found")
        
        # Check options analysis
        options_results = results.get('options_analysis', {})
        if options_results:
            logger.info(f"\n🎯 Options Analysis Results:")
            for index, analysis in options_results.items():
                if analysis:
                    logger.info(f"   {index}: Analysis available")
                    # Check for dummy data indicators
                    if 'dummy' in str(analysis).lower() or 'sample' in str(analysis).lower():
                        logger.warning(f"   ⚠️ {index} appears to use dummy data")
                else:
                    logger.warning(f"   ⚠️ {index}: No analysis data")
        else:
            logger.warning("⚠️ No options analysis results found")
        
        # Check market predictions
        predictions_results = results.get('market_predictions', {})
        if predictions_results:
            logger.info(f"\n🔮 Market Predictions Results:")
            for index, predictions in predictions_results.items():
                if predictions:
                    logger.info(f"   {index}: Predictions available")
                    # Check for dummy data indicators
                    if 'dummy' in str(predictions).lower() or 'sample' in str(predictions).lower():
                        logger.warning(f"   ⚠️ {index} appears to use dummy data")
                else:
                    logger.warning(f"   ⚠️ {index}: No prediction data")
        else:
            logger.warning("⚠️ No market predictions results found")
        
        # Test recommendations generation
        logger.info(f"\n🎯 Testing Trading Recommendations...")
        try:
            recommendations = screening_manager.generate_trading_recommendations(results)
            if recommendations:
                logger.info(f"✅ Recommendations generated successfully")
                logger.info(f"   High Confidence Signals: {len(recommendations.get('high_confidence_signals', []))}")
                logger.info(f"   Action Items: {len(recommendations.get('action_items', []))}")
                
                # Check for dummy data in recommendations
                if 'dummy' in str(recommendations).lower() or 'sample' in str(recommendations).lower():
                    logger.warning("⚠️ Recommendations appear to use dummy data")
            else:
                logger.warning("⚠️ No recommendations generated")
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
        
        # Save results for inspection
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"test_comprehensive_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to: {results_file}")
        
        # Summary
        logger.info(f"\n📋 SUMMARY:")
        logger.info("=" * 40)
        
        working_modules = []
        dummy_data_modules = []
        error_modules = []
        
        if eod_results:
            working_modules.append("EOD Screening")
        else:
            error_modules.append("EOD Screening")
        
        if intraday_results:
            working_modules.append("Intraday Screening")
        else:
            error_modules.append("Intraday Screening")
        
        if options_results:
            working_modules.append("Options Analysis")
            if 'dummy' in str(options_results).lower():
                dummy_data_modules.append("Options Analysis")
        else:
            error_modules.append("Options Analysis")
        
        if predictions_results:
            working_modules.append("Market Predictions")
            if 'dummy' in str(predictions_results).lower():
                dummy_data_modules.append("Market Predictions")
        else:
            error_modules.append("Market Predictions")
        
        logger.info(f"✅ Working Modules: {', '.join(working_modules) if working_modules else 'None'}")
        logger.info(f"⚠️ Dummy Data Modules: {', '.join(dummy_data_modules) if dummy_data_modules else 'None'}")
        logger.info(f"❌ Error Modules: {', '.join(error_modules) if error_modules else 'None'}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in comprehensive screening test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    test_comprehensive_screening()
