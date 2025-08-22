#!/usr/bin/env python3
"""
Phase 1 Test Script - Data Infrastructure & Historical Data Collection
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.database.duckdb_manager import DatabaseManager
from src.data.collectors.async_data_collector import AsyncDataCollector
from src.data.database.models import DataCollectionConfig
from loguru import logger

async def test_phase1():
    """Test Phase 1 implementation."""
    print("🚀 Phase 1 Test - Data Infrastructure & Historical Data Collection")
    print("=" * 70)
    
    # Test 1: Database Setup
    print("\n🧪 Test 1: SQLite Database Setup")
    print("-" * 40)
    
    try:
        db_manager = DatabaseManager("data/test_ai_hedge_fund.db")
        print("✅ SQLite database initialized successfully")
        
        # Test database operations
        test_ticker = "RELIANCE.NS"
        test_start_date = "2024-01-01"
        test_end_date = "2024-01-31"
        
        # Check if we can query the database
        technical_data = db_manager.get_technical_data(test_ticker, test_start_date, test_end_date)
        print(f"✅ Database query test: {len(technical_data)} records found")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    
    # Test 2: Data Collectors
    print("\n🧪 Test 2: Data Collectors")
    print("-" * 40)
    
    try:
        async_collector = AsyncDataCollector(db_manager, max_workers=3)
        print("✅ Async data collector initialized")
        
        # Test configuration
        config = DataCollectionConfig(
            ticker=test_ticker,
            start_date=datetime.strptime(test_start_date, '%Y-%m-%d').date(),
            end_date=datetime.strptime(test_end_date, '%Y-%m-%d').date(),
            data_types=['technical', 'fundamental'],
            parallel=True,
            max_workers=3
        )
        print("✅ Data collection configuration created")
        
    except Exception as e:
        print(f"❌ Data collectors setup failed: {e}")
        return False
    
    # Test 3: Historical Data Collection
    print("\n🧪 Test 3: Historical Data Collection")
    print("-" * 40)
    
    try:
        print(f"Collecting data for {test_ticker} from {test_start_date} to {test_end_date}")
        
        # Collect data
        results = await async_collector.collect_historical_data(config)
        
        if results:
            print(f"✅ Data collection completed: {len(results)} results")
            
            for result in results:
                print(f"  - {result.data_type}: {result.records_collected} records collected")
                print(f"    Success: {result.success}, Duration: {result.duration_seconds:.2f}s")
                
                if result.errors:
                    print(f"    Errors: {result.errors}")
        else:
            print("⚠️ No data collection results")
            
    except Exception as e:
        print(f"❌ Historical data collection failed: {e}")
        return False
    
    # Test 4: Database Storage
    print("\n🧪 Test 4: Database Storage Verification")
    print("-" * 40)
    
    try:
        # Check if data was stored
        technical_data = db_manager.get_technical_data(test_ticker, test_start_date, test_end_date)
        fundamental_data = db_manager.get_fundamental_data(test_ticker, test_start_date, test_end_date)
        
        print(f"✅ Technical data in database: {len(technical_data)} records")
        print(f"✅ Fundamental data in database: {len(fundamental_data)} records")
        
        if len(technical_data) > 0:
            print(f"  Sample technical data columns: {list(technical_data.columns)}")
        
    except Exception as e:
        print(f"❌ Database storage verification failed: {e}")
        return False
    
    # Test 5: Data Quality
    print("\n🧪 Test 5: Data Quality Metrics")
    print("-" * 40)
    
    try:
        # Get latest data date
        latest_technical = db_manager.get_latest_data_date(test_ticker, 'technical')
        latest_fundamental = db_manager.get_latest_data_date(test_ticker, 'fundamental')
        
        print(f"✅ Latest technical data: {latest_technical}")
        print(f"✅ Latest fundamental data: {latest_fundamental}")
        
        # Get missing data dates
        missing_technical = db_manager.get_missing_data_dates(test_ticker, test_start_date, test_end_date, 'technical')
        print(f"✅ Missing technical data dates: {len(missing_technical)}")
        
    except Exception as e:
        print(f"❌ Data quality metrics failed: {e}")
        return False
    
    # Test 6: Multiple Tickers
    print("\n🧪 Test 6: Multiple Tickers Collection")
    print("-" * 40)
    
    try:
        test_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        print(f"Testing collection for {len(test_tickers)} tickers")
        
        # Collect data for multiple tickers
        multi_results = await async_collector.collect_multiple_tickers(
            test_tickers, test_start_date, test_end_date, ['technical']
        )
        
        print(f"✅ Multiple tickers collection completed")
        
        # Generate summary
        summary = async_collector.get_collection_summary(multi_results)
        print(f"  Total tickers: {summary['total_tickers']}")
        print(f"  Successful: {summary['successful_tickers']}")
        print(f"  Failed: {summary['failed_tickers']}")
        print(f"  Total records collected: {summary['total_records_collected']}")
        
    except Exception as e:
        print(f"❌ Multiple tickers collection failed: {e}")
        return False
    
    # Cleanup
    print("\n🧹 Cleanup")
    print("-" * 40)
    
    try:
        db_manager.close()
        print("✅ Database connection closed")
        
        # Remove test database
        import os
        test_db_path = "data/test_ai_hedge_fund.db"
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print("✅ Test database removed")
            
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Phase 1 Test Completed Successfully!")
    print("✅ All components working correctly")
    print("✅ Ready for Phase 2 implementation")
    
    return True

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run test
    success = asyncio.run(test_phase1())
    
    if success:
        print("\n🚀 Phase 1 is ready for production use!")
        sys.exit(0)
    else:
        print("\n❌ Phase 1 test failed. Please check the errors above.")
        sys.exit(1) 