#!/usr/bin/env python3
"""
Debug migration to identify the issue
"""

import sqlite3
import duckdb
import pandas as pd
from loguru import logger

def debug_migration():
    """Debug migration to identify the issue."""
    logger.info("🔍 Debug migration to identify the issue...")
    
    try:
        # Read from SQLite backup
        logger.info("📖 Reading from SQLite backup...")
        sqlite_conn = sqlite3.connect('data/sqlite_backup/comprehensive_equity.db')
        
        # Get a small sample from SQLite
        sample_data = pd.read_sql_query("SELECT * FROM price_data LIMIT 5", sqlite_conn)
        logger.info(f"📊 Sample data from SQLite:")
        logger.info(f"   Columns: {list(sample_data.columns)}")
        logger.info(f"   Shape: {sample_data.shape}")
        logger.info(f"   Sample:\n{sample_data}")
        
        sqlite_conn.close()
        
        # Connect to DuckDB
        logger.info("📝 Connecting to DuckDB...")
        duckdb_conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Check DuckDB schema
        schema = duckdb_conn.execute("DESCRIBE price_data").fetchdf()
        logger.info(f"📋 DuckDB price_data schema:")
        logger.info(f"   {schema}")
        
        # Try to insert just one record first
        logger.info("🧪 Testing single record insert...")
        try:
            single_record = sample_data.iloc[0:1]
            logger.info(f"   Inserting: {single_record.to_dict('records')[0]}")
            
            # Convert to list of values
            values = single_record.values.tolist()[0]
            placeholders = ','.join(['?' for _ in values])
            
            duckdb_conn.execute(f"INSERT INTO price_data VALUES ({placeholders})", values)
            logger.info("   ✅ Single record insert successful")
            
            # Check if it was actually inserted
            count = duckdb_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
            logger.info(f"   📊 Records in table after insert: {count}")
            
        except Exception as e:
            logger.error(f"   ❌ Single record insert failed: {e}")
        
        duckdb_conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        return False

if __name__ == "__main__":
    debug_migration()
