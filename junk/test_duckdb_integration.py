#!/usr/bin/env python3
"""
Test DuckDB integration with comprehensive data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.tools.enhanced_api import get_prices
from src.data.providers.duckdb_provider import DuckDBProvider
from loguru import logger

def test_duckdb_integration():
    """Test DuckDB integration with comprehensive data."""
    logger.info("🧪 Testing DuckDB integration with comprehensive data...")
    
    try:
        # Test DuckDB provider directly
        logger.info("📊 Testing DuckDB provider directly...")
        duckdb_provider = DuckDBProvider()
        
        # Get database stats
        symbol_count = duckdb_provider.get_symbol_count()
        data_range = duckdb_provider.get_data_range()
        
        logger.info(f"✅ DuckDB Database Stats:")
        logger.info(f"   📈 Total symbols: {symbol_count:,}")
        logger.info(f"   📅 Date range: {data_range.get('start_date', 'N/A')} to {data_range.get('end_date', 'N/A')}")
        
        # Test with a few symbols
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        
        logger.info(f"🔍 Testing price data retrieval for {len(test_symbols)} symbols...")
        
        for symbol in test_symbols:
            try:
                # Test with DuckDB provider
                df = duckdb_provider.get_prices_as_dataframe(symbol, '2024-01-01', '2024-12-31')
                
                if not df.empty:
                    logger.info(f"   ✅ {symbol}: {len(df)} records retrieved")
                else:
                    logger.warning(f"   ⚠️ {symbol}: No data found")
                    
            except Exception as e:
                logger.error(f"   ❌ {symbol}: Error - {e}")
        
        # Test enhanced API integration
        logger.info("🔗 Testing enhanced API integration...")
        
        for symbol in test_symbols:
            try:
                # Test with enhanced API (should use DuckDB for Indian stocks)
                data = get_prices(f"{symbol}.NS", '2024-01-01', '2024-12-31')
                
                if isinstance(data, pd.DataFrame) and not data.empty:
                    logger.info(f"   ✅ {symbol}.NS via API: {len(data)} records")
                else:
                    logger.warning(f"   ⚠️ {symbol}.NS via API: No data found")
                    
            except Exception as e:
                logger.error(f"   ❌ {symbol}.NS via API: Error - {e}")
        
        logger.info("✅ DuckDB integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ DuckDB integration test failed: {e}")
        return False

def main():
    """Main function."""
    success = test_duckdb_integration()
    
    if success:
        logger.info("🎉 DuckDB integration test completed successfully!")
    else:
        logger.error("❌ DuckDB integration test failed.")

if __name__ == "__main__":
    main()
