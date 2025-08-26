#!/usr/bin/env python3
"""
Check Database Status
Checks the status of options chain database without locking it.
"""

import os
import sys
import time
from datetime import datetime

def check_database():
    """Check the options chain database status."""
    try:
        import duckdb
        
        # Try to connect to the database
        db_path = "data/options_chain_data.duckdb"
        
        if not os.path.exists(db_path):
            print("Database file does not exist")
            return
        
        # Get file modification time
        mod_time = os.path.getmtime(db_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Database file last modified: {mod_time_str}")
        
        # Try to connect and query
        try:
            conn = duckdb.connect(db_path)
            result = conn.execute('SELECT MAX(timestamp) as latest_time, COUNT(*) as total_records FROM options_chain_data').fetchall()
            conn.close()
            
            if result and result[0]:
                latest_time, total_records = result[0]
                print(f"Latest data timestamp: {latest_time}")
                print(f"Total records: {total_records}")
                
                # Calculate time difference
                if latest_time:
                    latest_dt = datetime.fromisoformat(str(latest_time).replace('Z', '+00:00'))
                    now = datetime.now()
                    time_diff = now - latest_dt.replace(tzinfo=None)
                    print(f"Time since last update: {time_diff}")
            else:
                print("No data found in database")
                
        except Exception as e:
            print(f"Could not query database (may be locked): {e}")
            
    except ImportError:
        print("DuckDB not available")
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_database()
