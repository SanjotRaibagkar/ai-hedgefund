#!/usr/bin/env python3
"""
Test script for Maintenance Scheduler
Tests the 6 AM daily update functionality
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append('./src')

from src.data.update.maintenance_scheduler import MaintenanceScheduler
from src.data.update.simple_price_updater import SimplePriceUpdater

def test_maintenance_scheduler():
    """Test the maintenance scheduler functionality."""
    print("🧪 Testing Maintenance Scheduler")
    print("=" * 50)
    
    try:
        # Initialize scheduler
        print("📋 Initializing Maintenance Scheduler...")
        scheduler = MaintenanceScheduler()
        
        # Get scheduler status
        print("📊 Getting scheduler status...")
        status = scheduler.get_scheduler_status()
        print(f"✅ Scheduler Status: {status.get('scheduler_running', False)}")
        print(f"📅 Configuration: {status.get('configuration', {})}")
        
        # Test daily update manually
        print("\n🔄 Testing daily update manually...")
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"📅 Target date: {yesterday}")
        
        # Run daily update
        scheduler._run_daily_update()
        
        print("✅ Daily update test completed")
        
        # Test EOD extra data update
        print("\n🔄 Testing EOD extra data update...")
        scheduler._run_eod_extra_data_update()
        
        print("✅ EOD extra data update test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_price_updater():
    """Test the simple price updater functionality."""
    print("\n🧪 Testing Simple Price Updater")
    print("=" * 50)
    
    try:
        # Initialize updater
        print("📋 Initializing Simple Price Updater...")
        updater = SimplePriceUpdater()
        
        # Get all symbols
        print("📊 Getting all symbols...")
        symbols = updater.get_all_symbols()
        print(f"✅ Found {len(symbols)} symbols")
        
        if symbols:
            # Test with first symbol
            test_symbol = symbols[0]
            print(f"🧪 Testing with symbol: {test_symbol}")
            
            # Get latest date
            latest_date = updater.get_latest_date_for_symbol(test_symbol)
            print(f"📅 Latest date for {test_symbol}: {latest_date}")
            
            # Test bhavcopy data
            if latest_date:
                print(f"📊 Testing bhavcopy data for {latest_date}...")
                bhavcopy_data = updater.get_bhavcopy_data(latest_date)
                if bhavcopy_data is not None and not bhavcopy_data.empty:
                    print(f"✅ Bhavcopy data: {len(bhavcopy_data)} records")
                else:
                    print("⚠️ No bhavcopy data available")
        
        updater.close()
        print("✅ Simple Price Updater test completed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_database_status():
    """Check the current database status."""
    print("\n📊 Checking Database Status")
    print("=" * 50)
    
    try:
        import duckdb
        
        # Connect to database
        conn = duckdb.connect("data/comprehensive_equity.duckdb")
        
        # Check price_data table
        print("📋 Checking price_data table...")
        result = conn.execute("SELECT COUNT(*) as total_records FROM price_data").fetchone()
        print(f"✅ Total price records: {result[0]}")
        
        # Check latest update
        result = conn.execute("SELECT MAX(last_updated) as latest_update FROM price_data").fetchone()
        print(f"📅 Latest update: {result[0]}")
        
        # Check unique symbols
        result = conn.execute("SELECT COUNT(DISTINCT symbol) as unique_symbols FROM price_data").fetchone()
        print(f"📊 Unique symbols: {result[0]}")
        
        # Check date range
        result = conn.execute("SELECT MIN(date) as earliest_date, MAX(date) as latest_date FROM price_data").fetchone()
        print(f"📅 Date range: {result[0]} to {result[1]}")
        
        conn.close()
        print("✅ Database status check completed")
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Maintenance Scheduler Test Suite")
    print("=" * 60)
    
    # Check database status
    db_ok = check_database_status()
    
    # Test simple price updater
    updater_ok = test_simple_price_updater()
    
    # Test maintenance scheduler
    scheduler_ok = test_maintenance_scheduler()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    print(f"Database Status: {'✅ PASS' if db_ok else '❌ FAIL'}")
    print(f"Simple Price Updater: {'✅ PASS' if updater_ok else '❌ FAIL'}")
    print(f"Maintenance Scheduler: {'✅ PASS' if scheduler_ok else '❌ FAIL'}")
    
    if db_ok and updater_ok and scheduler_ok:
        print("\n🎉 All tests passed! Maintenance scheduler is ready for 6 AM updates.")
        print("💡 Run 'setup_maintenance_scheduler.ps1' as Administrator to enable auto-startup.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
