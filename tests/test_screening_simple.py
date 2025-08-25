#!/usr/bin/env python3
"""
Simple Screening Test
Tests the screening system with the current data.
"""

import asyncio
import sqlite3
import time
from src.screening.duckdb_eod_screener import duckdb_eod_screener

async def test_screening():
    """Test the screening system."""
    print("🧪 TESTING SCREENING SYSTEM")
    print("=" * 40)
    
    # Initialize screener
    screener = duckdb_eod_screener
    
    # Get database stats
    print("📊 Database Statistics:")
    conn = sqlite3.connect("data/comprehensive_equity.db")
    
    total_symbols = conn.execute('SELECT COUNT(*) FROM securities').fetchone()[0]
    fno_symbols = conn.execute('SELECT COUNT(*) FROM securities WHERE is_fno = 1').fetchone()[0]
    total_records = conn.execute('SELECT COUNT(*) FROM price_data').fetchone()[0]
    
    print(f"   📈 Total Symbols: {total_symbols}")
    print(f"   🔥 FNO Symbols: {fno_symbols}")
    print(f"   📊 Total Records: {total_records:,}")
    
    # Get sample symbols
    print("\n🎯 Getting Sample Symbols...")
    
    # Get some major stocks
    major_stocks = conn.execute('''
        SELECT symbol FROM securities 
        WHERE symbol IN ('RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK')
        LIMIT 5
    ''').fetchall()
    
    symbols = [row[0] for row in major_stocks]
    print(f"   🎯 Testing symbols: {symbols}")
    
    conn.close()
    
    # Test screening
    print(f"\n🔍 Running EOD Screening...")
    start_time = time.time()
    
    try:
        # Run screening
        results = await screener.screen_universe(symbols=symbols)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Screening completed in {duration:.2f} seconds")
        
        # Display results
        print(f"\n📊 SCREENING RESULTS:")
        print("=" * 40)
        
        summary = results['summary']
        bullish = results['bullish_signals']
        bearish = results['bearish_signals']
        
        print(f"📈 Summary:")
        print(f"   🎯 Total Screened: {summary['total_screened']}")
        print(f"   ✅ Bullish Signals: {summary['bullish_signals']}")
        print(f"   ❌ Bearish Signals: {summary['bearish_signals']}")
        print(f"   📅 Date: {summary['screening_date']}")
        
        if bullish:
            print(f"\n📈 BULLISH Signals:")
            for signal in bullish[:3]:  # Show first 3
                print(f"   ✅ {signal['symbol']}: ₹{signal.get('current_price', 'N/A')} | Confidence: {signal.get('confidence', 'N/A')}%")
        
        if bearish:
            print(f"\n📉 BEARISH Signals:")
            for signal in bearish[:3]:  # Show first 3
                print(f"   ❌ {signal['symbol']}: ₹{signal.get('current_price', 'N/A')} | Confidence: {signal.get('confidence', 'N/A')}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during screening: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_connection():
    """Test database connection and data access."""
    print("🔍 Testing Database Connection...")
    
    try:
        conn = sqlite3.connect("data/comprehensive_equity.db")
        
        # Test basic queries
        total_symbols = conn.execute('SELECT COUNT(*) FROM securities').fetchone()[0]
        print(f"   ✅ Total symbols: {total_symbols}")
        
        # Test price data
        price_count = conn.execute('SELECT COUNT(*) FROM price_data').fetchone()[0]
        print(f"   ✅ Price records: {price_count:,}")
        
        # Test sample data
        sample_data = conn.execute('''
            SELECT symbol, date, close_price 
            FROM price_data 
            WHERE symbol = 'RELIANCE' 
            ORDER BY date DESC 
            LIMIT 5
        ''').fetchall()
        
        print(f"   ✅ Sample RELIANCE data: {len(sample_data)} records")
        for row in sample_data:
            print(f"      {row[0]} | {row[1]} | ₹{row[2]:.2f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING SCREENING TESTS")
    print("=" * 50)
    
    # Test database first
    db_ok = test_database_connection()
    
    if db_ok:
        # Test screening
        asyncio.run(test_screening())
    
    print(f"\n🎉 TESTING COMPLETED!")
    print("=" * 30)
