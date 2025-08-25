#!/usr/bin/env python3
"""
Test script for EOD Extra Data Downloader
Tests the functionality of downloading EOD extra data from NSE.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.downloaders.eod_extra_data_downloader import EODExtraDataDownloader
from src.data.update.daily_updater import DailyDataUpdater
from src.data.update.maintenance_scheduler import MaintenanceScheduler


def test_eod_extra_data_downloader():
    """Test the EOD Extra Data Downloader."""
    print("🧪 Testing EOD Extra Data Downloader")
    print("=" * 60)
    
    try:
        # Initialize downloader
        downloader = EODExtraDataDownloader()
        
        # Show current progress
        print("📊 Current Progress:")
        downloader.show_progress()
        
        # Get database stats
        print("\n📈 Current Database Stats:")
        stats = downloader.get_database_stats()
        if stats:
            for data_type, data_stats in stats.items():
                print(f"{data_type.upper()}:")
                print(f"  Total Records: {data_stats['total_records']:,}")
                print(f"  Unique Dates: {data_stats['unique_dates']}")
                if data_stats['earliest_date']:
                    print(f"  Date Range: {data_stats['earliest_date']} to {data_stats['latest_date']}")
                print()
        
        # Test downloading last 7 days of data
        print("🔄 Testing EOD Extra Data Download (Last 7 days)...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"Date Range: {start_date} to {end_date}")
        
        # Test individual downloads
        print("\n📥 Testing FNO Bhav Copy...")
        fno_result = downloader.download_fno_bhav_copy(start_date, end_date)
        print(f"FNO Result: {fno_result}")
        
        print("\n📥 Testing Equity Bhav Copy with Delivery...")
        equity_result = downloader.download_equity_bhav_copy_delivery(start_date, end_date)
        print(f"Equity Result: {equity_result}")
        
        print("\n📥 Testing Bhav Copy Indices...")
        indices_result = downloader.download_bhav_copy_indices(start_date, end_date)
        print(f"Indices Result: {indices_result}")
        
        print("\n📥 Testing FII DII Activity...")
        fii_result = downloader.download_fii_dii_activity(start_date, end_date)
        print(f"FII Result: {fii_result}")
        
        # Show updated stats
        print("\n📈 Updated Database Stats:")
        updated_stats = downloader.get_database_stats()
        if updated_stats:
            for data_type, data_stats in updated_stats.items():
                print(f"{data_type.upper()}:")
                print(f"  Total Records: {data_stats['total_records']:,}")
                print(f"  Unique Dates: {data_stats['unique_dates']}")
                if data_stats['earliest_date']:
                    print(f"  Date Range: {data_stats['earliest_date']} to {data_stats['latest_date']}")
                print()
        
        print("✅ EOD Extra Data Downloader test completed successfully!")
        
    except Exception as e:
        print(f"❌ EOD Extra Data Downloader test failed: {e}")
        import traceback
        traceback.print_exc()


def test_daily_updater():
    """Test the Daily Updater with EOD extra data."""
    print("\n🧪 Testing Daily Updater with EOD Extra Data")
    print("=" * 60)
    
    try:
        # Initialize daily updater
        updater = DailyDataUpdater()
        
        # Show update status
        print("📊 Update Status:")
        status = updater.get_update_status()
        print(f"Total Tickers: {status.get('total_tickers', 0)}")
        print(f"Indian Stocks: {status.get('indian_stocks', 0)}")
        print(f"US Stocks: {status.get('us_stocks', 0)}")
        print(f"Data Types: {status.get('data_types', [])}")
        print(f"EOD Extra Data Enabled: {status.get('eod_extra_data_enabled', False)}")
        print(f"EOD Extra Data Types: {status.get('eod_extra_data_types', [])}")
        
        # Test daily update for yesterday
        print("\n🔄 Testing Daily Update (Yesterday)...")
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        result = asyncio.run(updater.run_daily_update(yesterday))
        
        print(f"Daily Update Result: {result}")
        
        if result.get('success', False):
            summary = result.get('summary', {})
            print(f"\n📈 Summary:")
            print(f"  Total Tickers: {summary.get('total_tickers', 0)}")
            print(f"  Successful Tickers: {summary.get('successful_tickers', 0)}")
            print(f"  Failed Tickers: {summary.get('failed_tickers', 0)}")
            print(f"  EOD Data Types: {summary.get('eod_data_types', 0)}")
            print(f"  Failed EOD Types: {summary.get('failed_eod_types', 0)}")
        
        print("✅ Daily Updater test completed successfully!")
        
    except Exception as e:
        print(f"❌ Daily Updater test failed: {e}")
        import traceback
        traceback.print_exc()


def test_maintenance_scheduler():
    """Test the Maintenance Scheduler."""
    print("\n🧪 Testing Maintenance Scheduler")
    print("=" * 60)
    
    try:
        # Initialize scheduler
        scheduler = MaintenanceScheduler()
        
        # Show scheduler status
        print("📊 Scheduler Status:")
        status = scheduler.get_scheduler_status()
        print(f"Scheduler Running: {status.get('scheduler_running', False)}")
        print(f"Configuration: {status.get('configuration', {})}")
        
        # Test running jobs once
        print("\n🔄 Testing Job Execution...")
        
        # Test EOD extra data update
        print("Testing EOD Extra Data Update...")
        scheduler.run_once("eod_extra_data")
        
        # Test daily update
        print("Testing Daily Update...")
        scheduler.run_once("daily")
        
        print("✅ Maintenance Scheduler test completed successfully!")
        
    except Exception as e:
        print(f"❌ Maintenance Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("🧪 EOD Extra Data System Test Suite")
    print("=" * 80)
    
    # Test 1: EOD Extra Data Downloader
    test_eod_extra_data_downloader()
    
    # Test 2: Daily Updater
    test_daily_updater()
    
    # Test 3: Maintenance Scheduler
    test_maintenance_scheduler()
    
    print("\n🎉 All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
