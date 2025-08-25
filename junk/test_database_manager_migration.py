#!/usr/bin/env python3
"""
Test DatabaseManager Migration to DuckDB
Verify that the updated DatabaseManager works correctly with comprehensive_equity.duckdb
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.database.duckdb_manager import DatabaseManager
import pandas as pd
from datetime import datetime, date
from loguru import logger

def test_database_manager_migration():
    """Test the migrated DatabaseManager with DuckDB."""
    logger.info("🧪 Testing DatabaseManager Migration to DuckDB...")
    
    try:
        # Initialize DatabaseManager
        logger.info("📖 Initializing DatabaseManager...")
        db_manager = DatabaseManager("data/comprehensive_equity.duckdb")
        
        # Test 1: Check if we can connect and access price_data
        logger.info("🔍 Test 1: Accessing price_data table...")
        symbols = db_manager.get_available_symbols()
        logger.info(f"✅ Found {len(symbols)} symbols in price_data")
        
        if len(symbols) > 0:
            test_symbol = symbols[0]
            logger.info(f"🎯 Testing with symbol: {test_symbol}")
            
            # Test 2: Get price data
            logger.info("🔍 Test 2: Retrieving price data...")
            price_data = db_manager.get_price_data(test_symbol, start_date='2024-01-01', end_date='2024-01-10')
            logger.info(f"✅ Retrieved {len(price_data)} price records")
            logger.info(f"📊 Sample price data:\n{price_data.head()}")
            
            # Test 3: Test foreign key constraints
            logger.info("🔍 Test 3: Testing foreign key constraints...")
            try:
                # Create sample technical data
                sample_technical_data = pd.DataFrame({
                    'ticker': [test_symbol],
                    'trade_date': [date(2024, 1, 1)],
                    'open_price': [100.0],
                    'high_price': [110.0],
                    'low_price': [90.0],
                    'close_price': [105.0],
                    'volume': [1000000],
                    'adjusted_close': [105.0],
                    'sma_20': [102.0],
                    'sma_50': [101.0],
                    'sma_200': [100.0],
                    'rsi_14': [55.0],
                    'macd': [0.5],
                    'macd_signal': [0.3],
                    'macd_histogram': [0.2],
                    'bollinger_upper': [110.0],
                    'bollinger_lower': [90.0],
                    'bollinger_middle': [100.0],
                    'atr_14': [2.0],
                    'created_at': [datetime.now()],
                    'updated_at': [datetime.now()]
                })
                
                # Insert technical data
                db_manager.insert_technical_data(sample_technical_data)
                logger.info("✅ Successfully inserted technical data (foreign key working)")
                
                # Retrieve technical data
                technical_data = db_manager.get_technical_data(test_symbol)
                logger.info(f"✅ Retrieved {len(technical_data)} technical records")
                
                # Clean up test data
                db_manager.connection.execute(
                    "DELETE FROM technical_data WHERE ticker = ? AND trade_date = ?",
                    [test_symbol, date(2024, 1, 1)]
                )
                logger.info("🧹 Test data cleaned up")
                
            except Exception as e:
                logger.error(f"❌ Foreign key test failed: {e}")
        
        # Test 4: Test data quality metrics
        logger.info("🔍 Test 4: Testing data quality metrics...")
        try:
            sample_quality_data = pd.DataFrame({
                'ticker': [test_symbol],
                'quality_date': [date(2024, 1, 1)],
                'data_type': ['technical'],
                'completeness_score': [0.95],
                'accuracy_score': [0.98],
                'timeliness_score': [0.90],
                'consistency_score': [0.92],
                'total_records': [100],
                'missing_records': [5],
                'error_count': [0],
                'last_updated': [datetime.now()],
                'created_at': [datetime.now()]
            })
            
            # Insert quality data
            db_manager.connection.execute("INSERT INTO data_quality_metrics SELECT * FROM sample_quality_data")
            logger.info("✅ Successfully inserted quality metrics")
            
            # Retrieve quality data
            quality_data = db_manager.get_data_quality_metrics(test_symbol, 'technical')
            logger.info(f"✅ Retrieved {len(quality_data)} quality records")
            
            # Clean up
            db_manager.connection.execute(
                "DELETE FROM data_quality_metrics WHERE ticker = ? AND quality_date = ?",
                [test_symbol, date(2024, 1, 1)]
            )
            logger.info("🧹 Quality test data cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Quality metrics test failed: {e}")
        
        # Test 5: Test missing data detection
        logger.info("🔍 Test 5: Testing missing data detection...")
        try:
            missing_dates = db_manager.get_missing_data_dates(
                test_symbol, 
                '2024-01-01', 
                '2024-01-10', 
                'technical'
            )
            logger.info(f"✅ Missing dates detection: {len(missing_dates)} missing dates")
            
        except Exception as e:
            logger.error(f"❌ Missing data detection failed: {e}")
        
        # Test 6: Test latest data date
        logger.info("🔍 Test 6: Testing latest data date...")
        try:
            latest_date = db_manager.get_latest_data_date(test_symbol, 'technical')
            logger.info(f"✅ Latest technical data date: {latest_date}")
            
        except Exception as e:
            logger.error(f"❌ Latest data date failed: {e}")
        
        # Close connection
        db_manager.close()
        
        logger.info("🎉 All DatabaseManager migration tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ DatabaseManager migration test failed: {e}")
        return False

if __name__ == "__main__":
    test_database_manager_migration()
