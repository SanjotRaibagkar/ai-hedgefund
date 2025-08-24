#!/usr/bin/env python3
"""
Check all database files and their contents
"""

import duckdb
import os

def check_database(db_path):
    """Check a single database file."""
    try:
        conn = duckdb.connect(db_path)
        
        # Check if price_data table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        if 'price_data' not in table_names:
            print(f"❌ {db_path}: No price_data table found")
            return
        
        # Get price data stats
        result = conn.execute("""
            SELECT 
                COUNT(*) as records,
                COUNT(DISTINCT symbol) as symbols,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM price_data
        """).fetchone()
        
        print(f"✅ {db_path}:")
        print(f"   Records: {result[0]:,}")
        print(f"   Symbols: {result[1]:,}")
        print(f"   Date Range: {result[2]} to {result[3]}")
        
        # Show sample symbols
        symbols = conn.execute("SELECT DISTINCT symbol FROM price_data LIMIT 5").fetchall()
        symbol_list = [s[0] for s in symbols]
        print(f"   Sample symbols: {', '.join(symbol_list)}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ {db_path}: Error - {e}")

def main():
    """Main function."""
    print("🔍 CHECKING ALL DATABASE FILES")
    print("=" * 60)
    
    # List all database files
    db_files = [f for f in os.listdir('data') if f.endswith(('.db', '.duckdb'))]
    
    for db_file in db_files:
        db_path = os.path.join('data', db_file)
        check_database(db_path)
        print()

if __name__ == "__main__":
    main()
