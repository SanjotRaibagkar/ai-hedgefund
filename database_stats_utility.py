#!/usr/bin/env python3
"""
Database Statistics Utility
Comprehensive tool to analyze all databases and show detailed statistics.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
import glob

def get_database_files():
    """Get all database files in the data directory."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    
    db_files = glob.glob(os.path.join(data_dir, "*.db"))
    return db_files

def analyze_database(db_path):
    """Analyze a single database and return comprehensive statistics."""
    print(f"\n🔍 ANALYZING DATABASE: {os.path.basename(db_path)}")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get table information
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"📊 Tables found: {table_names}")
        
        for table_name in table_names:
            print(f"\n📋 TABLE: {table_name}")
            print("-" * 40)
            
            # Get table schema
            schema = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            print("📐 SCHEMA:")
            for col in schema:
                col_id, name, data_type, not_null, default_val, pk = col
                print(f"  • {name}: {data_type} {'(PRIMARY KEY)' if pk else ''}")
            
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"📊 Total rows: {row_count:,}")
            
            # Get sample data
            sample_data = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
            if sample_data:
                print("📄 Sample data:")
                for row in sample_data:
                    print(f"  • {row}")
            
            # If it's a price_data table, get detailed statistics
            if table_name == 'price_data':
                analyze_price_data(conn, table_name)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error analyzing {db_path}: {e}")

def analyze_price_data(conn, table_name):
    """Analyze price data table specifically."""
    print(f"\n📈 PRICE DATA ANALYSIS:")
    print("-" * 30)
    
    # Get unique symbols
    symbols = conn.execute(f"SELECT DISTINCT symbol FROM {table_name}").fetchall()
    unique_symbols = [s[0] for s in symbols]
    print(f"🎯 Unique symbols: {len(unique_symbols)}")
    
    # Show sample symbols
    if unique_symbols:
        print(f"📋 Sample symbols: {unique_symbols[:10]}")
    
    # Get date range
    date_range = conn.execute(f"""
        SELECT MIN(date) as start_date, MAX(date) as end_date 
        FROM {table_name}
    """).fetchone()
    
    if date_range[0] and date_range[1]:
        print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
        
        # Calculate days
        try:
            start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
            end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
            days = (end_date - start_date).days + 1
            print(f"📊 Total days: {days}")
        except:
            pass
    
    # Get records per symbol
    records_per_symbol = conn.execute(f"""
        SELECT symbol, COUNT(*) as record_count 
        FROM {table_name} 
        GROUP BY symbol 
        ORDER BY record_count DESC
    """).fetchall()
    
    if records_per_symbol:
        print(f"\n📊 Records per symbol (top 10):")
        for symbol, count in records_per_symbol[:10]:
            print(f"  • {symbol}: {count} records")
    
    # Get price statistics
    price_stats = conn.execute(f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT symbol) as unique_symbols,
            AVG(close_price) as avg_price,
            MIN(close_price) as min_price,
            MAX(close_price) as max_price
        FROM {table_name}
        WHERE close_price > 0
    """).fetchone()
    
    if price_stats:
        print(f"\n💰 Price Statistics:")
        print(f"  • Total records: {price_stats[0]:,}")
        print(f"  • Unique symbols: {price_stats[1]}")
        print(f"  • Average price: ₹{price_stats[2]:.2f}")
        print(f"  • Price range: ₹{price_stats[3]:.2f} - ₹{price_stats[4]:.2f}")

def show_comprehensive_stats():
    """Show comprehensive statistics for all databases."""
    print("🎯 COMPREHENSIVE DATABASE STATISTICS")
    print("=" * 70)
    
    # Get all database files
    db_files = get_database_files()
    
    if not db_files:
        print("❌ No database files found in data/ directory")
        return
    
    print(f"📁 Found {len(db_files)} database file(s):")
    for db_file in db_files:
        print(f"  • {os.path.basename(db_file)}")
    
    # Analyze each database
    for db_file in db_files:
        analyze_database(db_file)
    
    # Show summary
    print(f"\n🎉 SUMMARY:")
    print("=" * 30)
    print(f"📊 Total databases analyzed: {len(db_files)}")
    
    # Check for screening results
    csv_files = glob.glob("*.csv")
    if csv_files:
        print(f"📄 CSV files found: {len(csv_files)}")
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"  • {csv_file}: {len(df)} records")
            except:
                print(f"  • {csv_file}: Error reading")

def show_detailed_symbol_analysis(db_path):
    """Show detailed analysis for specific symbols."""
    print(f"\n🔍 DETAILED SYMBOL ANALYSIS")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get all symbols with their data
        symbols_data = conn.execute("""
            SELECT symbol, COUNT(*) as records, 
                   MIN(date) as first_date, MAX(date) as last_date,
                   AVG(close_price) as avg_price
            FROM price_data 
            GROUP BY symbol 
            ORDER BY records DESC
        """).fetchall()
        
        print(f"📊 Detailed analysis for {len(symbols_data)} symbols:")
        print("-" * 50)
        
        for symbol, records, first_date, last_date, avg_price in symbols_data[:20]:
            print(f"🎯 {symbol}:")
            print(f"  • Records: {records}")
            print(f"  • Date range: {first_date} to {last_date}")
            print(f"  • Average price: ₹{avg_price:.2f}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error in detailed analysis: {e}")

def main():
    """Main function."""
    print("🎯 DATABASE STATISTICS UTILITY")
    print("=" * 50)
    
    # Show comprehensive stats
    show_comprehensive_stats()
    
    # Show detailed analysis for the main database
    main_db = "data/final_comprehensive.db"
    if os.path.exists(main_db):
        show_detailed_symbol_analysis(main_db)
    
    print(f"\n🎉 DATABASE ANALYSIS COMPLETED!")
    print("📊 All databases have been analyzed and statistics displayed.")

if __name__ == "__main__":
    main() 