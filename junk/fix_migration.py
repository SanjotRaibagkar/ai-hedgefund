#!/usr/bin/env python3
"""
Fixed migration from SQLite to DuckDB
Handle schema differences and properly migrate data
"""

import sqlite3
import duckdb
import pandas as pd
from loguru import logger

def fix_migration():
    """Fixed migration from SQLite to DuckDB."""
    logger.info("🚀 Fixed migration from SQLite to DuckDB...")
    
    try:
        # Read from SQLite backup
        logger.info("📖 Reading from SQLite backup...")
        sqlite_conn = sqlite3.connect('data/sqlite_backup/comprehensive_equity.db')
        
        # Check SQLite schema
        logger.info("🔍 Checking SQLite schema...")
        sqlite_schema = sqlite_conn.execute("PRAGMA table_info(price_data)").fetchall()
        logger.info(f"📋 SQLite price_data columns: {[col[1] for col in sqlite_schema]}")
        
        # Read price_data from SQLite
        price_data = pd.read_sql_query("SELECT * FROM price_data", sqlite_conn)
        logger.info(f"✅ Read {len(price_data):,} price records from SQLite")
        logger.info(f"📊 Price data columns: {list(price_data.columns)}")
        logger.info(f"📊 Price data sample:\n{price_data.head(2)}")
        
        # Check if securities table exists and read it
        tables = sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [table[0] for table in tables]
        
        securities_data = None
        if 'securities' in table_names:
            securities_data = pd.read_sql_query("SELECT * FROM securities", sqlite_conn)
            logger.info(f"✅ Read {len(securities_data):,} securities records from SQLite")
            logger.info(f"📊 Securities columns: {list(securities_data.columns)}")
        
        sqlite_conn.close()
        
        # Connect to DuckDB and check schema
        logger.info("🔍 Checking DuckDB schema...")
        duckdb_conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        duckdb_schema = duckdb_conn.execute("DESCRIBE price_data").fetchdf()
        logger.info(f"📋 DuckDB price_data columns: {duckdb_schema['column_name'].tolist()}")
        
        # Clear existing data
        duckdb_conn.execute("DELETE FROM price_data")
        if securities_data is not None:
            duckdb_conn.execute("DELETE FROM securities")
        
        # Insert price_data - handle schema differences
        logger.info("📝 Inserting price_data to DuckDB...")
        
        # Check if we need to handle the 'id' column
        if 'id' in price_data.columns and 'id' in duckdb_schema['column_name'].tolist():
            # Both have id column, insert normally
            duckdb_conn.execute("INSERT INTO price_data SELECT * FROM price_data")
        elif 'id' in price_data.columns and 'id' not in duckdb_schema['column_name'].tolist():
            # SQLite has id but DuckDB doesn't, exclude id column
            columns_to_insert = [col for col in price_data.columns if col != 'id']
            price_data_subset = price_data[columns_to_insert]
            duckdb_conn.execute("INSERT INTO price_data SELECT * FROM price_data_subset")
        else:
            # No id column issue, insert normally
            duckdb_conn.execute("INSERT INTO price_data SELECT * FROM price_data")
        
        logger.info(f"✅ Inserted {len(price_data):,} price records to DuckDB")
        
        # Insert securities if available
        if securities_data is not None:
            duckdb_conn.execute("INSERT INTO securities SELECT * FROM securities")
            logger.info(f"✅ Inserted {len(securities_data):,} securities records to DuckDB")
        
        # Verify the migration
        price_count = duckdb_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        symbol_count = duckdb_conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        
        logger.info(f"✅ Migration complete!")
        logger.info(f"   📊 Price records: {price_count:,}")
        logger.info(f"   🏢 Unique symbols: {symbol_count:,}")
        
        if securities_data is not None:
            securities_count = duckdb_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
            logger.info(f"   📋 Securities records: {securities_count:,}")
        
        # Test foreign key constraints
        logger.info("🔗 Testing foreign key constraints...")
        try:
            sample_symbol = duckdb_conn.execute("SELECT symbol FROM price_data LIMIT 1").fetchone()[0]
            sample_date = duckdb_conn.execute("SELECT date FROM price_data LIMIT 1").fetchone()[0]
            
            logger.info(f"   🎯 Testing with symbol: {sample_symbol}, date: {sample_date}")
            
            duckdb_conn.execute("""
                INSERT INTO technical_data (ticker, trade_date, open_price, high_price, low_price, close_price, volume, created_at, updated_at)
                VALUES (?, ?, 100.0, 110.0, 90.0, 105.0, 1000000, NOW(), NOW())
            """, [sample_symbol, sample_date])
            logger.info(f"   ✅ Foreign key constraint working")
            
            # Clean up test data
            duckdb_conn.execute("DELETE FROM technical_data WHERE ticker = ? AND trade_date = ?", [sample_symbol, sample_date])
            logger.info(f"   🧹 Test data cleaned up")
            
        except Exception as e:
            logger.error(f"   ❌ Foreign key constraint failed: {e}")
        
        duckdb_conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    fix_migration()
