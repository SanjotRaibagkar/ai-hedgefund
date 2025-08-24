#!/usr/bin/env python3
"""
Test Optimized Equity Data Downloader with Delta Updates
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.downloaders.optimized_equity_downloader import OptimizedEquityDataDownloader
from datetime import datetime, timedelta
import duckdb

def test_delta_update():
    """Test delta update functionality - check latest data and download missing data."""
    print("🎯 TESTING OPTIMIZED EQUITY DATA DOWNLOADER WITH DELTA UPDATES")
    print("=" * 70)
    
    # Initialize downloader
    downloader = OptimizedEquityDataDownloader()
    
    # Show current progress
    downloader.show_progress()
    
    # Get current database stats
    stats = downloader.get_database_stats()
    if stats:
        print(f"\n📊 CURRENT DATABASE STATS:")
        print(f"Securities: {stats['securities_count']}")
        print(f"Active securities: {stats['active_securities']}")
        print(f"Total records: {stats['total_records']:,}")
        print(f"Unique symbols: {stats['unique_symbols']}")
        if stats['date_range'][0]:
            print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"Avg records per symbol: {stats['avg_records_per_symbol']:.1f}")
    
    # Check latest data in database
    print(f"\n🔍 CHECKING LATEST DATA IN DATABASE...")
    latest_db_date = downloader.get_latest_data_date()
    print(f"Latest data in database: {latest_db_date}")
    
    # Get current date (latest available data)
    current_date = datetime.now().date()
    print(f"Current date: {current_date}")
    
    # Check if we need delta update
    if latest_db_date and latest_db_date < current_date:
        days_missing = (current_date - latest_db_date).days
        print(f"⚠️  Database is {days_missing} days behind current date")
        print(f"🔄 Starting delta update to download missing data...")
        
        # Download delta data
        success = downloader.download_delta_data(latest_db_date, current_date)
        
        if success:
            print(f"✅ Delta update completed successfully!")
            
            # Show updated stats
            updated_stats = downloader.get_database_stats()
            if updated_stats:
                print(f"\n📊 UPDATED DATABASE STATS:")
                print(f"Total records: {updated_stats['total_records']:,}")
                if updated_stats['date_range'][1]:
                    print(f"Latest date: {updated_stats['date_range'][1]}")
        else:
            print(f"❌ Delta update failed!")
    else:
        print(f"✅ Database is up to date! Latest data: {latest_db_date}")
    
    # Test with a few specific symbols
    print(f"\n🧪 TESTING DELTA UPDATE FOR SPECIFIC SYMBOLS...")
    test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
    
    for symbol in test_symbols:
        print(f"\n📈 Testing {symbol}:")
        symbol_latest = downloader.get_symbol_latest_date(symbol)
        print(f"   Latest data for {symbol}: {symbol_latest}")
        
        if symbol_latest and symbol_latest < current_date:
            print(f"   ⚠️  {symbol} needs update")
            success = downloader.update_symbol_data(symbol, symbol_latest, current_date)
            if success:
                print(f"   ✅ {symbol} updated successfully")
            else:
                print(f"   ❌ {symbol} update failed")
        else:
            print(f"   ✅ {symbol} is up to date")
    
    print(f"\n🎉 DELTA UPDATE TEST COMPLETED!")

def test_initial_download_and_delta():
    """Test initial download and then delta updates."""
    print("🎯 TESTING INITIAL DOWNLOAD AND DELTA UPDATES")
    print("=" * 70)
    
    # Initialize downloader
    downloader = OptimizedEquityDataDownloader()
    
    # Show current progress
    downloader.show_progress()
    
    # Get current database stats
    stats = downloader.get_database_stats()
    if stats:
        print(f"\n📊 CURRENT DATABASE STATS:")
        print(f"Securities: {stats['securities_count']}")
        print(f"Active securities: {stats['active_securities']}")
        print(f"Total records: {stats['total_records']:,}")
        print(f"Unique symbols: {stats['unique_symbols']}")
        if stats['date_range'][0]:
            print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"Avg records per symbol: {stats['avg_records_per_symbol']:.1f}")
    
    # If database is empty, do initial download
    if stats['total_records'] == 0:
        print(f"\n🔄 DATABASE IS EMPTY - DOING INITIAL DOWNLOAD...")
        print(f"Downloading data for 5 test symbols...")
        
        # Download initial data for a few symbols
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        success_count = 0
        
        for symbol in test_symbols:
            print(f"\n📈 Downloading {symbol}...")
            try:
                # Get current price info
                current_data = downloader.nse.price_info(symbol)
                if current_data and current_data.get('LastTradedPrice', 0) > 0:
                    # Generate historical data for last 30 days
                    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    to_date = datetime.now().strftime('%Y-%m-%d')
                    
                    historical_data = downloader._simulate_historical_data(symbol, from_date, to_date, current_data)
                    if historical_data:
                        downloader._store_symbol_delta_data(symbol, historical_data)
                        print(f"   ✅ {symbol}: {len(historical_data)} records")
                        success_count += 1
                    else:
                        print(f"   ❌ {symbol}: No data generated")
                else:
                    print(f"   ❌ {symbol}: No current data available")
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
        
        print(f"\n📊 INITIAL DOWNLOAD COMPLETED: {success_count}/{len(test_symbols)} symbols")
    
    # Now test delta updates
    print(f"\n🔍 TESTING DELTA UPDATES...")
    latest_db_date = downloader.get_latest_data_date()
    current_date = datetime.now().date()
    
    print(f"Latest data in database: {latest_db_date}")
    print(f"Current date: {current_date}")
    
    if latest_db_date and latest_db_date < current_date:
        days_missing = (current_date - latest_db_date).days
        print(f"⚠️  Database is {days_missing} days behind current date")
        print(f"🔄 Starting delta update...")
        
        # Test delta update for one symbol
        test_symbol = 'RELIANCE'
        symbol_latest = downloader.get_symbol_latest_date(test_symbol)
        print(f"\n📈 Testing delta update for {test_symbol}:")
        print(f"   Latest data: {symbol_latest}")
        
        if symbol_latest and symbol_latest < current_date:
            print(f"   ⚠️  {test_symbol} needs update")
            # Convert datetime.date to string format
            from_date_str = symbol_latest.strftime('%Y-%m-%d') if hasattr(symbol_latest, 'strftime') else str(symbol_latest)
            to_date_str = current_date.strftime('%Y-%m-%d')
            success = downloader.update_symbol_data(test_symbol, from_date_str, to_date_str)
            if success:
                print(f"   ✅ {test_symbol} delta update successful")
            else:
                print(f"   ❌ {test_symbol} delta update failed")
        else:
            print(f"   ✅ {test_symbol} is up to date")
    else:
        print(f"✅ Database is up to date!")
    
    # Show final stats
    final_stats = downloader.get_database_stats()
    if final_stats:
        print(f"\n📊 FINAL DATABASE STATS:")
        print(f"Securities: {final_stats['securities_count']}")
        print(f"Total records: {final_stats['total_records']:,}")
        print(f"Unique symbols: {final_stats['unique_symbols']}")
        if final_stats['date_range'][1]:
            print(f"Latest date: {final_stats['date_range'][1]}")
    
    print(f"\n🎉 INITIAL DOWNLOAD AND DELTA TEST COMPLETED!")

def main():
    """Test the optimized equity data downloader with delta updates."""
    test_initial_download_and_delta()

if __name__ == "__main__":
    main() 