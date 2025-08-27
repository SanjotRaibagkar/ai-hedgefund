#!/usr/bin/env python3
"""
Test Individual Modules
Check each module separately to identify what's working and what's using dummy data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
import json
from datetime import datetime

def test_eod_screening():
    """Test EOD screening module."""
    logger.info("🧪 Testing EOD Screening Module")
    logger.info("=" * 40)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        screening_manager = ScreeningManager()
        
        # Test with a few stocks
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        results = screening_manager.get_eod_signals(test_stocks)
        
        logger.info(f"✅ EOD Screening Results:")
        logger.info(f"   Bullish: {len(results.get('bullish_signals', []))}")
        logger.info(f"   Bearish: {len(results.get('bearish_signals', []))}")
        
        if results.get('bullish_signals'):
            sample = results['bullish_signals'][0]
            logger.info(f"   Sample Signal: {sample.get('symbol', 'N/A')} - Confidence: {sample.get('confidence', 'N/A')}%")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ EOD Screening Error: {e}")
        return None

def test_intraday_screening():
    """Test Intraday screening module."""
    logger.info("🧪 Testing Intraday Screening Module")
    logger.info("=" * 40)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        screening_manager = ScreeningManager()
        
        # Test with a few stocks
        test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        results = screening_manager.intraday_screener.screen_stocks(test_stocks)
        
        logger.info(f"✅ Intraday Screening Results:")
        logger.info(f"   Breakout: {len(results.get('breakout_signals', []))}")
        logger.info(f"   Reversal: {len(results.get('reversal_signals', []))}")
        logger.info(f"   Momentum: {len(results.get('momentum_signals', []))}")
        
        if results.get('breakout_signals'):
            sample = results['breakout_signals'][0]
            logger.info(f"   Sample Breakout: {sample.get('symbol', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Intraday Screening Error: {e}")
        return None

def test_options_analysis():
    """Test Options analysis module."""
    logger.info("🧪 Testing Options Analysis Module")
    logger.info("=" * 40)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        screening_manager = ScreeningManager()
        
        # Test NIFTY analysis
        nifty_results = screening_manager.options_analyzer.analyze_index_options('NIFTY')
        
        logger.info(f"✅ NIFTY Options Analysis:")
        logger.info(f"   Current Price: {nifty_results.get('current_price', 'N/A')}")
        logger.info(f"   PCR: {nifty_results.get('analysis', {}).get('oi_analysis', {}).get('put_call_ratio', 'N/A')}")
        
        # Check for dummy data indicators
        if 'dummy' in str(nifty_results).lower() or 'sample' in str(nifty_results).lower():
            logger.warning("⚠️ NIFTY analysis appears to use dummy data")
        else:
            logger.info("✅ NIFTY analysis uses real data")
        
        return nifty_results
        
    except Exception as e:
        logger.error(f"❌ Options Analysis Error: {e}")
        return None

def test_market_predictions():
    """Test Market predictions module."""
    logger.info("🧪 Testing Market Predictions Module")
    logger.info("=" * 40)
    
    try:
        from src.screening.screening_manager import ScreeningManager
        screening_manager = ScreeningManager()
        
        # Test NIFTY predictions
        nifty_prediction = screening_manager.market_predictor.predict_market_movement('NIFTY', 'eod')
        
        logger.info(f"✅ NIFTY Market Prediction:")
        logger.info(f"   Direction: {nifty_prediction.get('prediction', {}).get('direction', 'N/A')}")
        logger.info(f"   Confidence: {nifty_prediction.get('prediction', {}).get('confidence', 'N/A')}%")
        
        # Check for dummy data indicators
        if 'dummy' in str(nifty_prediction).lower() or 'sample' in str(nifty_prediction).lower():
            logger.warning("⚠️ NIFTY predictions appear to use dummy data")
        else:
            logger.info("✅ NIFTY predictions use real data")
        
        return nifty_prediction
        
    except Exception as e:
        logger.error(f"❌ Market Predictions Error: {e}")
        return None

def test_database_connectivity():
    """Test database connectivity and data availability."""
    logger.info("🧪 Testing Database Connectivity")
    logger.info("=" * 40)
    
    try:
        from src.data.database.duckdb_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        # Test getting available symbols
        symbols = db_manager.get_available_symbols()
        logger.info(f"✅ Available symbols: {len(symbols)}")
        
        # Test getting price data for a few symbols
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
        
        for symbol in test_symbols:
            try:
                price_data = db_manager.get_price_data(symbol, days=30)
                if price_data is not None and len(price_data) > 0:
                    logger.info(f"✅ {symbol}: {len(price_data)} price records available")
                else:
                    logger.warning(f"⚠️ {symbol}: No price data available")
            except Exception as e:
                logger.error(f"❌ {symbol}: Error getting price data - {e}")
        
        return symbols
        
    except Exception as e:
        logger.error(f"❌ Database Connectivity Error: {e}")
        return None

def main():
    """Run all tests."""
    logger.info("🚀 Starting Individual Module Tests")
    logger.info("=" * 60)
    
    # Test database connectivity first
    test_database_connectivity()
    logger.info("")
    
    # Test EOD screening
    test_eod_screening()
    logger.info("")
    
    # Test intraday screening
    test_intraday_screening()
    logger.info("")
    
    # Test options analysis
    test_options_analysis()
    logger.info("")
    
    # Test market predictions
    test_market_predictions()
    logger.info("")
    
    logger.info("🏁 All tests completed!")

if __name__ == "__main__":
    main()
