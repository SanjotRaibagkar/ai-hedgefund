#!/usr/bin/env python3
"""
Simple migration from SQLite to DuckDB
"""

import sqlite3
import duckdb
import pandas as pd
from loguru import logger

def migrate_data():
    """Migrate data from SQLite to DuckDB."""
    logger.info("🚀 Starting simple migration...")
    
    try:
        # Read from SQLite
        sqlite_conn = sqlite3.connect('data/comprehensive_equity.db')
        price_data = pd.read_sql_query("SELECT * FROM price_data", sqlite_conn)
        securities_data = pd.read_sql_query("SELECT * FROM securities", sqlite_conn)
        sqlite_conn.close()
        
        logger.info(f"📊 Read {len(price_data):,} price records and {len(securities_data):,} securities")
        
        # Write to DuckDB
        duckdb_conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Clear existing data
        duckdb_conn.execute("DELETE FROM price_data")
        duckdb_conn.execute("DELETE FROM securities")
        
        # Insert using pandas
        duckdb_conn.execute("INSERT INTO price_data SELECT * FROM price_data")
        duckdb_conn.execute("INSERT INTO securities SELECT * FROM securities")
        
        # Verify
        price_count = duckdb_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        symbol_count = duckdb_conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        
        logger.info(f"✅ Migration complete: {price_count:,} records, {symbol_count:,} symbols")
        duckdb_conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    migrate_data()
