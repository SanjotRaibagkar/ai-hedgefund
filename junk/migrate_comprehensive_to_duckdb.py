#!/usr/bin/env python3
"""
Migrate comprehensive data from SQLite to DuckDB
"""

import sqlite3
import duckdb
import pandas as pd
from loguru import logger

def migrate_comprehensive_data():
    """Migrate comprehensive data from SQLite to DuckDB."""
    logger.info("🚀 Starting comprehensive data migration to DuckDB...")
    
    try:
        # Connect to SQLite database
        sqlite_conn = sqlite3.connect('data/comprehensive_equity.db')
        
        # Connect to DuckDB database
        duckdb_conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Get data from SQLite
        logger.info("📊 Reading data from SQLite...")
        price_data = pd.read_sql_query("SELECT * FROM price_data", sqlite_conn)
        securities_data = pd.read_sql_query("SELECT * FROM securities", sqlite_conn)
        
        logger.info(f"✅ Read {len(price_data):,} price records and {len(securities_data):,} securities from SQLite")
        
        # Clear existing data in DuckDB
        logger.info("🧹 Clearing existing data in DuckDB...")
        duckdb_conn.execute("DELETE FROM price_data")
        duckdb_conn.execute("DELETE FROM securities")
        
        # Insert data into DuckDB using pandas
        logger.info("📥 Inserting price data into DuckDB...")
        duckdb_conn.execute("DELETE FROM price_data")
        duckdb_conn.execute("INSERT INTO price_data SELECT * FROM price_data")
        
        logger.info("📥 Inserting securities data into DuckDB...")
        duckdb_conn.execute("DELETE FROM securities")
        duckdb_conn.execute("INSERT INTO securities SELECT * FROM securities")
        
        # Verify the migration
        logger.info("🔍 Verifying migration...")
        price_count = duckdb_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        symbol_count = duckdb_conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        securities_count = duckdb_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        
        logger.info(f"✅ Migration completed successfully!")
        logger.info(f"   📊 Price records: {price_count:,}")
        logger.info(f"   🏢 Unique symbols: {symbol_count:,}")
        logger.info(f"   📋 Securities records: {securities_count:,}")
        
        # Close connections
        sqlite_conn.close()
        duckdb_conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

def main():
    """Main function."""
    success = migrate_comprehensive_data()
    
    if success:
        logger.info("🎉 Comprehensive data migration to DuckDB completed successfully!")
        logger.info("✅ All strategies will now use the comprehensive DuckDB database")
    else:
        logger.error("❌ Migration failed. Please check the errors above.")

if __name__ == "__main__":
    main()
