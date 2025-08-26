#!/usr/bin/env python3
"""
Check Intraday Data Collection Status
Quick script to check if intraday data collection is working.
"""

import duckdb
from datetime import datetime

def check_intraday_status():
    """Check the status of intraday data collection."""
    print("📊 Intraday Data Collection Status Check")
    print("=" * 50)
    
    try:
        # Connect to the intraday database
        conn = duckdb.connect('data/intraday_ml_data.duckdb')
        
        # Check tables
        tables = conn.execute('SHOW TABLES').fetchall()
        print(f"📋 Database Tables: {len(tables)}")
        for table in tables:
            print(f"   - {table[0]}")
        
        print()
        
        # Check options data
        options_count = conn.execute('SELECT COUNT(*) FROM intraday_options_data').fetchall()[0][0]
        latest_options = conn.execute('SELECT MAX(timestamp) FROM intraday_options_data').fetchall()[0][0]
        
        print(f"📈 Options Data:")
        print(f"   Total Records: {options_count:,}")
        print(f"   Latest Update: {latest_options}")
        
        # Check index data
        index_count = conn.execute('SELECT COUNT(*) FROM intraday_index_data').fetchall()[0][0]
        latest_index = conn.execute('SELECT MAX(timestamp) FROM intraday_index_data').fetchall()[0][0]
        
        print(f"📊 Index Data:")
        print(f"   Total Records: {index_count:,}")
        print(f"   Latest Update: {latest_index}")
        
        # Check recent activity
        print(f"\n🕒 Recent Activity (Last 5 entries):")
        recent_data = conn.execute('''
            SELECT timestamp, index_symbol, spot_price, COUNT(*) as records 
            FROM intraday_options_data 
            GROUP BY timestamp, index_symbol, spot_price 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''').fetchall()
        
        for row in recent_data:
            timestamp, index_symbol, spot_price, records = row
            print(f"   {timestamp} | {index_symbol} | ₹{spot_price:,.2f} | {records} records")
        
        # Check if data is being collected today
        today = datetime.now().date()
        today_data = conn.execute('''
            SELECT COUNT(*) 
            FROM intraday_options_data 
            WHERE DATE(timestamp) = ?
        ''', [today]).fetchall()[0][0]
        
        print(f"\n📅 Today's Data Collection:")
        print(f"   Records Today: {today_data:,}")
        
        if today_data > 0:
            print("   ✅ Data collection is active today!")
        else:
            print("   ⚠️ No data collected today")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking intraday status: {e}")

if __name__ == "__main__":
    check_intraday_status()
