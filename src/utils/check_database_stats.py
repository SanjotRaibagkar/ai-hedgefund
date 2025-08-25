#!/usr/bin/env python3
"""
Check Database Statistics
Shows how many symbols were downloaded and stored in the database.
"""

import asyncio
import sys
import os
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.enhanced_indian_data_manager import enhanced_indian_data_manager

def main():
    """Check database statistics."""
    print("📊 Database Statistics")
    print("=" * 50)
    
    try:
        # Get database stats
        stats = enhanced_indian_data_manager.get_database_stats()
        
        print(f"📈 Total Securities: {stats['total_securities']}")
        print(f"📊 Total Data Points: {stats['total_data_points']}")
        print(f"📅 Date Range: {stats['date_range']}")
        print(f"💾 Database Size: {stats['database_size_mb']:.2f} MB")
        print(f"🔗 NSEUtility Available: {stats['nse_utility_available']}")
        
        # Get sample symbols
        print("\n📋 Sample Securities in Database:")
        print("-" * 40)
        
        # Note: Securities table was removed, using price_data instead
        symbols = enhanced_indian_data_manager.db_manager.get_available_symbols()[:10]
        symbols = [(symbol, symbol) for symbol in symbols]  # Convert to (symbol, name) format
            
            for symbol, name in symbols:
                print(f"  • {symbol}: {name}")
        
        # Get data points per symbol
        if stats['total_securities'] > 0:
            avg_data_points = stats['total_data_points'] // stats['total_securities']
            print(f"\n📊 Average Data Points per Symbol: {avg_data_points}")
        
        # Get detailed symbol data
        print("\n📈 Detailed Symbol Data:")
        print("-" * 40)
        
        # Get symbol counts from DuckDB
        symbol_counts = enhanced_indian_data_manager.db_manager.connection.execute("""
            SELECT symbol, COUNT(*) as data_points 
            FROM price_data 
            GROUP BY symbol 
            ORDER BY data_points DESC
        """).fetchall()
            
            for symbol, count in symbol_counts:
                print(f"  • {symbol}: {count} data points")
        
        print("\n✅ Database check completed!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 