#!/usr/bin/env python3
"""
Check DuckDB Tables Structure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import duckdb

def check_duckdb_tables():
    """Check DuckDB table structure."""
    conn = duckdb.connect('data/optimized_equity.duckdb')
    
    print("🗄️ DUCKDB TABLE ANALYSIS")
    print("=" * 50)
    
    # Show all tables
    tables = conn.execute('SHOW TABLES').fetchall()
    print(f"📋 Available Tables: {len(tables)}")
    for table in tables:
        print(f"  - {table[0]}")
    
    print("\n🔍 TABLE SCHEMAS:")
    print("=" * 50)
    
    # Show schema for each table
    for table in tables:
        table_name = table[0]
        print(f"\n📊 {table_name.upper()} TABLE:")
        print("-" * 30)
        
        # Get schema
        schema = conn.execute(f'DESCRIBE {table_name}').fetchdf()
        for _, row in schema.iterrows():
            print(f"  {row['column_name']:<20} {row['column_type']:<15} {row['null']}")
        
        # Get record count
        count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
        print(f"\n  📈 Records: {count:,}")
        
        # Show sample data
        if count > 0:
            print(f"  📄 Sample Data:")
            sample = conn.execute(f'SELECT * FROM {table_name} LIMIT 3').fetchdf()
            print(sample.to_string(max_cols=8, max_colwidth=15))
    
    # Detailed price_data analysis
    print("\n" + "="*60)
    print("📊 DETAILED PRICE_DATA TABLE STATISTICS")
    print("="*60)
    
    # Basic stats
    total_records = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
    unique_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
    
    print(f"📈 Total Records: {total_records:,}")
    print(f"🎯 Unique Symbols: {unique_symbols}")
    
    # Date range
    date_range = conn.execute("SELECT MIN(date), MAX(date) FROM price_data").fetchone()
    print(f"📅 Date Range: {date_range[0]} to {date_range[1]}")
    
    # Records per symbol
    print(f"\n📊 RECORDS PER SYMBOL:")
    symbol_counts = conn.execute("""
        SELECT symbol, COUNT(*) as record_count, 
               MIN(date) as first_date, MAX(date) as last_date
        FROM price_data 
        GROUP BY symbol 
        ORDER BY record_count DESC
    """).fetchdf()
    
    for _, row in symbol_counts.iterrows():
        print(f"  {row['symbol']:<12} {row['record_count']:>3} records  ({row['first_date']} to {row['last_date']})")
    
    # Price statistics
    print(f"\n💰 PRICE STATISTICS:")
    price_stats = conn.execute("""
        SELECT 
            AVG(close_price) as avg_close,
            MIN(close_price) as min_close,
            MAX(close_price) as max_close,
            AVG(volume) as avg_volume,
            SUM(volume) as total_volume
        FROM price_data
    """).fetchone()
    
    print(f"  Average Close Price: ₹{price_stats[0]:.2f}")
    print(f"  Min Close Price: ₹{price_stats[1]:.2f}")
    print(f"  Max Close Price: ₹{price_stats[2]:.2f}")
    print(f"  Average Volume: {price_stats[3]:,.0f}")
    print(f"  Total Volume: {price_stats[4]:,.0f}")
    
    # Latest data for each symbol
    print(f"\n🕒 LATEST DATA BY SYMBOL:")
    latest_data = conn.execute("""
        SELECT symbol, MAX(date) as latest_date, 
               ANY_VALUE(close_price) as latest_close,
               ANY_VALUE(volume) as latest_volume
        FROM price_data 
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) 
            FROM price_data 
            GROUP BY symbol
        )
        GROUP BY symbol
        ORDER BY symbol
    """).fetchdf()
    
    for _, row in latest_data.iterrows():
        print(f"  {row['symbol']:<12} {row['latest_date']}  ₹{row['latest_close']:.2f}  Vol: {row['latest_volume']:,}")
    
    # Data completeness check
    print(f"\n✅ DATA COMPLETENESS:")
    expected_trading_days = conn.execute("""
        SELECT COUNT(DISTINCT date) as trading_days
        FROM price_data
    """).fetchone()[0]
    
    print(f"  Trading Days Covered: {expected_trading_days}")
    print(f"  Average Records per Symbol: {total_records/unique_symbols:.1f}")
    
    # Missing data analysis
    print(f"\n⚠️  MISSING DATA ANALYSIS:")
    missing_data = conn.execute("""
        SELECT symbol, 
               COUNT(*) as actual_records,
               (SELECT COUNT(DISTINCT date) FROM price_data) as total_trading_days,
               (SELECT COUNT(DISTINCT date) FROM price_data) - COUNT(*) as missing_records
        FROM price_data 
        GROUP BY symbol
        HAVING COUNT(*) < (SELECT COUNT(DISTINCT date) FROM price_data)
        ORDER BY missing_records DESC
    """).fetchdf()
    
    if len(missing_data) > 0:
        for _, row in missing_data.iterrows():
            print(f"  {row['symbol']:<12} Missing {row['missing_records']} records")
    else:
        print("  ✅ All symbols have complete data!")
    
    conn.close()

if __name__ == "__main__":
    check_duckdb_tables()
