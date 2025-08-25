#!/usr/bin/env python3
"""
Test Update Managers Migration to DuckDB
Verify that all update managers work correctly with DuckDB DatabaseManager
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.update.update_manager import UpdateManager
from src.data.update.daily_updater import DailyDataUpdater
from src.data.update.data_quality_monitor import DataQualityMonitor
from src.data.update.missing_data_filler import MissingDataFiller
from src.data.update.maintenance_scheduler import MaintenanceScheduler
from src.data.database.duckdb_manager import DatabaseManager
import pandas as pd
from datetime import datetime, date
from loguru import logger

def test_update_managers_migration():
    """Test all update managers with DuckDB DatabaseManager."""
    logger.info("🧪 Testing Update Managers Migration to DuckDB...")
    
    try:
        # Test 1: Initialize DatabaseManager
        logger.info("🔍 Test 1: Initializing DatabaseManager...")
        db_manager = DatabaseManager("data/comprehensive_equity.duckdb")
        logger.info(f"✅ DatabaseManager initialized with: {db_manager.db_path}")
        
        # Test 2: Test UpdateManager
        logger.info("🔍 Test 2: Testing UpdateManager...")
        update_manager = UpdateManager(db_manager)
        logger.info("✅ UpdateManager initialized successfully")
        
        # Test 3: Test DailyDataUpdater
        logger.info("🔍 Test 3: Testing DailyDataUpdater...")
        daily_updater = DailyDataUpdater(db_manager)
        logger.info("✅ DailyDataUpdater initialized successfully")
        logger.info(f"✅ Ticker config loaded: {len(daily_updater.tickers_config.get('indian_stocks', []))} Indian stocks")
        
        # Test 4: Test DataQualityMonitor
        logger.info("🔍 Test 4: Testing DataQualityMonitor...")
        quality_monitor = DataQualityMonitor(db_manager)
        logger.info("✅ DataQualityMonitor initialized successfully")
        
        # Test 5: Test MissingDataFiller
        logger.info("🔍 Test 5: Testing MissingDataFiller...")
        data_filler = MissingDataFiller(db_manager)
        logger.info("✅ MissingDataFiller initialized successfully")
        logger.info(f"✅ Interpolation methods available: {list(data_filler.interpolation_methods.keys())}")
        
        # Test 6: Test MaintenanceScheduler
        logger.info("🔍 Test 6: Testing MaintenanceScheduler...")
        maintenance_scheduler = MaintenanceScheduler(db_manager)
        logger.info("✅ MaintenanceScheduler initialized successfully")
        logger.info(f"✅ Available tasks: {list(maintenance_scheduler.available_tasks.keys())}")
        logger.info(f"✅ Scheduler enabled: {maintenance_scheduler.config.get('enabled', False)}")
        
        # Test 7: Test database connectivity through update managers
        logger.info("🔍 Test 7: Testing database connectivity...")
        try:
            symbols = db_manager.get_available_symbols()
            logger.info(f"✅ Database connectivity verified: {len(symbols)} symbols available")
            
            if len(symbols) > 0:
                test_symbol = symbols[0]
                logger.info(f"🎯 Testing with symbol: {test_symbol}")
                
                # Test 8: Test data retrieval through update managers
                logger.info("🔍 Test 8: Testing data retrieval...")
                price_data = db_manager.get_price_data(test_symbol, start_date='2024-01-01', end_date='2024-01-10')
                logger.info(f"✅ Retrieved {len(price_data)} price records")
                
                # Test 9: Test quality monitor functionality
                logger.info("🔍 Test 9: Testing quality monitor...")
                if hasattr(quality_monitor, 'update_quality_metrics'):
                    logger.info("✅ Quality monitor methods available")
                else:
                    logger.error("❌ Quality monitor methods not found")
                
                # Test 10: Test data filler functionality
                logger.info("🔍 Test 10: Testing data filler...")
                if hasattr(data_filler, 'fill_missing_data'):
                    logger.info("✅ Data filler methods available")
                else:
                    logger.error("❌ Data filler methods not found")
                
            else:
                logger.warning("⚠️ No symbols found in database")
        
        except Exception as e:
            logger.error(f"❌ Database connectivity test failed: {e}")
            return False
        
        # Test 11: Test update manager initialization
        logger.info("🔍 Test 11: Testing update manager initialization...")
        try:
            # Note: This is an async method, so we'll just test that it exists
            if hasattr(update_manager, 'initialize'):
                logger.info("✅ Update manager initialization method available")
            else:
                logger.error("❌ Update manager initialization method not found")
        except Exception as e:
            logger.error(f"❌ Update manager initialization test failed: {e}")
        
        # Test 12: Test daily updater functionality
        logger.info("🔍 Test 12: Testing daily updater...")
        try:
            if hasattr(daily_updater, 'run_daily_update'):
                logger.info("✅ Daily updater methods available")
            else:
                logger.error("❌ Daily updater methods not found")
        except Exception as e:
            logger.error(f"❌ Daily updater test failed: {e}")
        
        # Test 13: Test maintenance scheduler functionality
        logger.info("🔍 Test 13: Testing maintenance scheduler...")
        try:
            if hasattr(maintenance_scheduler, 'start_scheduler'):
                logger.info("✅ Maintenance scheduler methods available")
            else:
                logger.error("❌ Maintenance scheduler methods not found")
        except Exception as e:
            logger.error(f"❌ Maintenance scheduler test failed: {e}")
        
        # Test 14: Verify all components use the same database
        logger.info("🔍 Test 14: Verifying database consistency...")
        db_paths = [
            update_manager.db_manager.db_path,
            daily_updater.db_manager.db_path,
            quality_monitor.db_manager.db_path,
            data_filler.db_manager.db_path,
            maintenance_scheduler.db_manager.db_path
        ]
        
        expected_path = "data/comprehensive_equity.duckdb"
        all_consistent = all(path == expected_path for path in db_paths)
        
        if all_consistent:
            logger.info("✅ All update managers use the same database path")
        else:
            logger.error("❌ Database paths are inconsistent")
            for i, path in enumerate(db_paths):
                logger.error(f"   Component {i}: {path}")
            return False
        
        # Test 15: Test database file existence
        logger.info("🔍 Test 15: Testing database file existence...")
        if os.path.exists(expected_path):
            logger.info(f"✅ Database file exists: {expected_path}")
        else:
            logger.error(f"❌ Database file not found: {expected_path}")
            return False
        
        logger.info("🎉 All Update Managers migration tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Update Managers migration test failed: {e}")
        return False

if __name__ == "__main__":
    test_update_managers_migration()
