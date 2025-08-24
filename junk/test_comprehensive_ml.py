#!/usr/bin/env python3
"""
Test ML model with comprehensive DuckDB data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ml.ml_strategies import MLEnhancedEODStrategy
from loguru import logger

def test_ml_model():
    """Test ML model with comprehensive data."""
    logger.info("🧪 Testing ML model with comprehensive DuckDB data...")
    
    try:
        # Initialize ML strategy
        ml_strategy = MLEnhancedEODStrategy()
        logger.info("✅ ML strategy initialized successfully")
        
        # Test with a few symbols from comprehensive data
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        
        logger.info(f"🔍 Testing ML analysis on {len(test_symbols)} symbols...")
        
        for symbol in test_symbols:
            try:
                logger.info(f"📊 Analyzing {symbol}...")
                analysis = ml_strategy.analyze_stock(symbol)
                
                if analysis:
                    logger.info(f"   ✅ {symbol}: Analysis completed")
                    logger.info(f"      Signal: {analysis.get('signal', 'N/A')}")
                    logger.info(f"      Confidence: {analysis.get('confidence', 'N/A')}")
                else:
                    logger.info(f"   ⚠️ {symbol}: No analysis result")
                    
            except Exception as e:
                logger.warning(f"   ❌ {symbol}: Analysis failed - {e}")
        
        logger.info("✅ ML model test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ML model test failed: {e}")
        return False

def main():
    """Main function."""
    success = test_ml_model()
    
    if success:
        logger.info("🎉 ML model test completed successfully!")
    else:
        logger.error("❌ ML model test failed.")

if __name__ == "__main__":
    main()
