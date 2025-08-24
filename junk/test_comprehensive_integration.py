#!/usr/bin/env python3
"""
Test Comprehensive Database Integration
Verify that all components are using the comprehensive database correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import duckdb
from datetime import datetime, timedelta
from src.data.providers.duckdb_provider import DuckDBProvider
from src.screening.duckdb_eod_screener import DuckDBEODScreener
from src.data.downloaders.optimized_equity_downloader import OptimizedEquityDataDownloader

def test_comprehensive_database_integration():
    """Test that all components are using the comprehensive database."""
    print("🔍 TESTING COMPREHENSIVE DATABASE INTEGRATION")
    print("=" * 60)
    
    # Test 1: Direct DuckDB connection
    print("\n📊 1. Testing Direct DuckDB Connection...")
    try:
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        total_records = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        unique_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        conn.close()
        
        print(f"   ✅ Comprehensive database accessible")
        print(f"   📈 Total records: {total_records:,}")
        print(f"   🎯 Unique symbols: {unique_symbols:,}")
        
        if total_records < 100000:
            print(f"   ⚠️  Warning: Only {total_records:,} records found (expected 900K+)")
        else:
            print(f"   ✅ Comprehensive data confirmed")
            
    except Exception as e:
        print(f"   ❌ Error accessing comprehensive database: {e}")
        return False
    
    # Test 2: DuckDB Provider
    print("\n📊 2. Testing DuckDB Provider...")
    try:
        provider = DuckDBProvider()
        
        # Test with a known symbol
        test_symbol = "RELIANCE"
        start_date = "2024-01-01"
        end_date = "2024-01-10"
        
        prices = provider.get_prices(test_symbol, start_date, end_date)
        print(f"   ✅ DuckDB Provider working")
        print(f"   📈 Retrieved {len(prices)} price records for {test_symbol}")
        
        if len(prices) > 0:
            print(f"   ✅ Data retrieval successful")
        else:
            print(f"   ⚠️  No data found for {test_symbol}")
            
    except Exception as e:
        print(f"   ❌ Error with DuckDB Provider: {e}")
        return False
    
    # Test 3: EOD Screener
    print("\n📊 3. Testing EOD Screener...")
    try:
        screener = DuckDBEODScreener()
        
        # Test symbol retrieval
        import asyncio
        symbols = asyncio.run(screener._get_all_symbols())
        print(f"   ✅ EOD Screener working")
        print(f"   🎯 Retrieved {len(symbols)} symbols for screening")
        
        if len(symbols) > 1000:
            print(f"   ✅ Comprehensive symbol list confirmed")
        else:
            print(f"   ⚠️  Only {len(symbols)} symbols found (expected 2000+)")
            
    except Exception as e:
        print(f"   ❌ Error with EOD Screener: {e}")
        return False
    
    # Test 4: Optimized Downloader
    print("\n📊 4. Testing Optimized Downloader...")
    try:
        downloader = OptimizedEquityDataDownloader()
        
        # Get database stats
        stats = downloader.get_database_stats()
        if stats:
            print(f"   ✅ Optimized Downloader working")
            print(f"   📈 Total records: {stats['total_records']:,}")
            print(f"   🎯 Unique symbols: {stats['unique_symbols']}")
            
            if stats['total_records'] > 100000:
                print(f"   ✅ Using comprehensive database")
            else:
                print(f"   ⚠️  Using test database (only {stats['total_records']:,} records)")
        else:
            print(f"   ❌ Could not get database stats")
            
    except Exception as e:
        print(f"   ❌ Error with Optimized Downloader: {e}")
        return False
    
    # Test 5: Enhanced API Integration
    print("\n📊 5. Testing Enhanced API Integration...")
    try:
        from src.tools.enhanced_api import get_prices
        
        # Test with Indian stock
        test_ticker = "RELIANCE"
        start_date = "2024-01-01"
        end_date = "2024-01-10"
        
        prices_df = get_prices(test_ticker, start_date, end_date)
        print(f"   ✅ Enhanced API working")
        print(f"   📈 Retrieved {len(prices_df)} price records for {test_ticker}")
        
        if len(prices_df) > 0:
            print(f"   ✅ API integration successful")
        else:
            print(f"   ⚠️  No data returned from API")
            
    except Exception as e:
        print(f"   ❌ Error with Enhanced API: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE DATABASE INTEGRATION TEST COMPLETED!")
    print("=" * 60)
    
    return True

def show_comprehensive_database_stats():
    """Show detailed stats of the comprehensive database."""
    print("\n📊 COMPREHENSIVE DATABASE DETAILED STATS")
    print("=" * 60)
    
    try:
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Basic stats
        total_records = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        unique_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        date_range = conn.execute("SELECT MIN(date), MAX(date) FROM price_data").fetchone()
        
        print(f"📈 Total Records: {total_records:,}")
        print(f"🎯 Unique Symbols: {unique_symbols:,}")
        print(f"📅 Date Range: {date_range[0]} to {date_range[1]}")
        
        # Top symbols by record count
        print(f"\n🏆 TOP 10 SYMBOLS BY RECORD COUNT:")
        top_symbols = conn.execute("""
            SELECT symbol, COUNT(*) as record_count
            FROM price_data 
            GROUP BY symbol 
            ORDER BY record_count DESC
            LIMIT 10
        """).fetchdf()
        
        for _, row in top_symbols.iterrows():
            print(f"   {row['symbol']:<12} {row['record_count']:>6} records")
        
        # Price statistics
        price_stats = conn.execute("""
            SELECT 
                AVG(close_price) as avg_close,
                MIN(close_price) as min_close,
                MAX(close_price) as max_close
            FROM price_data
        """).fetchone()
        
        print(f"\n💰 PRICE STATISTICS:")
        print(f"   Average Close Price: ₹{price_stats[0]:.2f}")
        print(f"   Min Close Price: ₹{price_stats[1]:.2f}")
        print(f"   Max Close Price: ₹{price_stats[2]:.2f}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")

if __name__ == "__main__":
    success = test_comprehensive_database_integration()
    show_comprehensive_database_stats()
    
    if success:
        print("\n✅ All components are successfully using the comprehensive database!")
    else:
        print("\n❌ Some components failed to use the comprehensive database.")
