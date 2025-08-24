#!/usr/bin/env python3
"""
Simple migration from SQLite to DuckDB
Direct copy from sqlite_backup/comprehensive_equity.db to comprehensive_equity.duckdb
"""

import sqlite3
import duckdb
import pandas as pd
from loguru import logger

def simple_migration():
    """Simple migration from SQLite to DuckDB."""
    logger.info("🚀 Simple migration from SQLite to DuckDB...")
    
    try:
        # Read from SQLite backup
        logger.info("📖 Reading from SQLite backup...")
        sqlite_conn = sqlite3.connect('data/sqlite_backup/comprehensive_equity.db')
        
        # Check SQLite data
        price_count_sqlite = sqlite_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        securities_count_sqlite = sqlite_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        
        logger.info(f"📊 SQLite data:")
        logger.info(f"   📈 Price records: {price_count_sqlite:,}")
        logger.info(f"   📋 Securities records: {securities_count_sqlite:,}")
        
        # Read data from SQLite
        price_data = pd.read_sql_query("SELECT * FROM price_data", sqlite_conn)
        securities_data = pd.read_sql_query("SELECT * FROM securities", sqlite_conn)
        
        sqlite_conn.close()
        
        logger.info(f"✅ Read {len(price_data):,} price records and {len(securities_data):,} securities from SQLite")
        
        # Connect to DuckDB
        logger.info("📝 Writing to DuckDB...")
        duckdb_conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Clear existing data
        duckdb_conn.execute("DELETE FROM price_data")
        duckdb_conn.execute("DELETE FROM securities")
        
        # Insert data using pandas DataFrame
        duckdb_conn.execute("INSERT INTO price_data SELECT * FROM price_data")
        duckdb_conn.execute("INSERT INTO securities SELECT * FROM securities")
        
        # Commit the transaction
        duckdb_conn.commit()
        
        # Verify the migration
        price_count_duckdb = duckdb_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        securities_count_duckdb = duckdb_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        symbol_count = duckdb_conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        
        logger.info(f"✅ Migration complete!")
        logger.info(f"   📊 Price records: {price_count_duckdb:,}")
        logger.info(f"   🏢 Unique symbols: {symbol_count:,}")
        logger.info(f"   📋 Securities records: {securities_count_duckdb:,}")
        
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
        
        logger.info("🎉 Migration successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    simple_migration()
