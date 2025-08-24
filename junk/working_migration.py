#!/usr/bin/env python3
"""
Working migration from SQLite to DuckDB
"""

import sqlite3
import duckdb
import pandas as pd
from loguru import logger

def working_migration():
    """Working migration from SQLite to DuckDB."""
    logger.info("🚀 Working migration from SQLite to DuckDB...")
    
    try:
        # Read from SQLite
        logger.info("📖 Reading from SQLite...")
        sqlite_conn = sqlite3.connect('data/comprehensive_equity.db')
        price_data = pd.read_sql_query("SELECT * FROM price_data", sqlite_conn)
        securities_data = pd.read_sql_query("SELECT * FROM securities", sqlite_conn)
        sqlite_conn.close()
        
        logger.info(f"✅ Read {len(price_data):,} price records and {len(securities_data):,} securities from SQLite")
        
        # Write to DuckDB
        logger.info("📝 Writing to DuckDB...")
        duckdb_conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Clear existing data
        duckdb_conn.execute("DELETE FROM price_data")
        duckdb_conn.execute("DELETE FROM securities")
        
        # Insert using pandas DataFrame directly
        duckdb_conn.execute("INSERT INTO price_data SELECT * FROM price_data")
        duckdb_conn.execute("INSERT INTO securities SELECT * FROM securities")
        
        # Verify
        price_count = duckdb_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        symbol_count = duckdb_conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        securities_count = duckdb_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        
        logger.info(f"✅ Migration complete!")
        logger.info(f"   📊 Price records: {price_count:,}")
        logger.info(f"   🏢 Unique symbols: {symbol_count:,}")
        logger.info(f"   📋 Securities records: {securities_count:,}")
        
        duckdb_conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    working_migration()
