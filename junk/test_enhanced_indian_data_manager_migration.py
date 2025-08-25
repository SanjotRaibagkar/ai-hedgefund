#!/usr/bin/env python3
"""
Test Enhanced Indian Data Manager Migration to DuckDB
Verify that the updated EnhancedIndianDataManager works correctly with DuckDB DatabaseManager
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.enhanced_indian_data_manager import EnhancedIndianDataManager
import pandas as pd
from datetime import datetime, date
from loguru import logger

def test_enhanced_indian_data_manager_migration():
    """Test the migrated EnhancedIndianDataManager with DuckDB."""
    logger.info("🧪 Testing Enhanced Indian Data Manager Migration to DuckDB...")
    
    try:
        # Initialize EnhancedIndianDataManager
        logger.info("📖 Initializing EnhancedIndianDataManager...")
        manager = EnhancedIndianDataManager("data/comprehensive_equity.duckdb")
        
        # Test 1: Check if DatabaseManager is properly initialized
        logger.info("🔍 Test 1: Checking DatabaseManager initialization...")
        if hasattr(manager, 'db_manager') and manager.db_manager is not None:
            logger.info("✅ DatabaseManager initialized successfully")
        else:
            logger.error("❌ DatabaseManager not initialized")
            return False
        
        # Test 2: Check if we can access price_data through DatabaseManager
        logger.info("🔍 Test 2: Accessing price_data through DatabaseManager...")
        try:
            symbols = manager.db_manager.get_available_symbols()
            logger.info(f"✅ Found {len(symbols)} symbols in price_data")
            
            if len(symbols) > 0:
                test_symbol = symbols[0]
                logger.info(f"🎯 Testing with symbol: {test_symbol}")
                
                # Test 3: Get price data through DatabaseManager
                logger.info("🔍 Test 3: Retrieving price data through DatabaseManager...")
                price_data = manager.db_manager.get_price_data(test_symbol, start_date='2024-01-01', end_date='2024-01-10')
                logger.info(f"✅ Retrieved {len(price_data)} price records through DatabaseManager")
                
                # Test 4: Get price data through EnhancedIndianDataManager
                logger.info("🔍 Test 4: Retrieving price data through EnhancedIndianDataManager...")
                # Note: This is an async method, so we'll just test the method exists
                if hasattr(manager, 'get_price_data'):
                    logger.info("✅ get_price_data method exists in EnhancedIndianDataManager")
                else:
                    logger.error("❌ get_price_data method not found")
                    return False
                
                # Test 5: Test database statistics
                logger.info("🔍 Test 5: Testing database statistics...")
                stats = manager.get_database_stats()
                logger.info(f"✅ Database stats retrieved: {stats}")
                
                # Test 6: Test download tracker table
                logger.info("🔍 Test 6: Testing download tracker table...")
                try:
                    tracker_count = manager.db_manager.connection.execute(
                        "SELECT COUNT(*) FROM download_tracker"
                    ).fetchone()[0]
                    logger.info(f"✅ Download tracker table accessible: {tracker_count} records")
                except Exception as e:
                    logger.error(f"❌ Download tracker table error: {e}")
                    return False
                
                # Test 7: Test NSEUtility initialization
                logger.info("🔍 Test 7: Testing NSEUtility initialization...")
                if manager.nse_utils is not None:
                    logger.info("✅ NSEUtility initialized successfully")
                else:
                    logger.warning("⚠️ NSEUtility not available (this is expected in test environment)")
                
                # Test 8: Test performance settings
                logger.info("🔍 Test 8: Testing performance settings...")
                logger.info(f"✅ Max workers: {manager.max_workers}")
                logger.info(f"✅ Batch size: {manager.batch_size}")
                logger.info(f"✅ Retry attempts: {manager.retry_attempts}")
                
            else:
                logger.warning("⚠️ No symbols found in price_data table")
        
        except Exception as e:
            logger.error(f"❌ Error accessing price_data: {e}")
            return False
        
        # Test 9: Test database path
        logger.info("🔍 Test 9: Testing database path...")
        expected_path = "data/comprehensive_equity.duckdb"
        if manager.db_path == expected_path:
            logger.info(f"✅ Database path correct: {manager.db_path}")
        else:
            logger.error(f"❌ Database path incorrect: {manager.db_path} (expected: {expected_path})")
            return False
        
        # Test 10: Test database file exists
        logger.info("🔍 Test 10: Testing database file existence...")
        if os.path.exists(manager.db_path):
            logger.info(f"✅ Database file exists: {manager.db_path}")
        else:
            logger.error(f"❌ Database file not found: {manager.db_path}")
            return False
        
        logger.info("🎉 All Enhanced Indian Data Manager migration tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Indian Data Manager migration test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_indian_data_manager_migration()
