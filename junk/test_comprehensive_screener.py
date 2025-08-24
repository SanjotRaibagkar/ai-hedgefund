#!/usr/bin/env python3
"""
Test EOD screener with comprehensive DuckDB data
"""

import asyncio
from src.screening.duckdb_eod_screener import DuckDBEODScreener
from loguru import logger

async def test_eod_screener():
    """Test EOD screener with comprehensive data."""
    logger.info("🧪 Testing EOD screener with comprehensive DuckDB data...")
    
    try:
        screener = DuckDBEODScreener()
        logger.info("✅ Screener initialized successfully")
        
        # Test screening
        logger.info("🔍 Running screening on comprehensive data...")
        results = await screener.screen_universe()
        
        logger.info(f"✅ Screening completed!")
        logger.info(f"   📊 Total signals generated: {len(results)}")
        
        if results:
            logger.info("📋 Sample signals:")
            for i, (symbol, signal) in enumerate(list(results.items())[:10]):
                logger.info(f"   {i+1}. {symbol}: {signal}")
        else:
            logger.info("   ℹ️ No signals generated (this is normal with limited historical data)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Screener test failed: {e}")
        return False

def main():
    """Main function."""
    success = asyncio.run(test_eod_screener())
    
    if success:
        logger.info("🎉 EOD screener test completed successfully!")
    else:
        logger.error("❌ EOD screener test failed.")

if __name__ == "__main__":
    main()
