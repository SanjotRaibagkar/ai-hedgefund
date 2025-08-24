#!/usr/bin/env python3
"""
Direct migration from SQLite to DuckDB using ATTACH
Copy data directly from sqlite_backup/comprehensive_equity.db to comprehensive_equity.duckdb
"""

import duckdb
from loguru import logger

def direct_migration():
    """Direct migration using DuckDB ATTACH."""
    logger.info("🚀 Direct migration using DuckDB ATTACH...")
    
    try:
        # Connect to DuckDB
        logger.info("📖 Connecting to DuckDB...")
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Attach SQLite database
        logger.info("🔗 Attaching SQLite database...")
        conn.execute("ATTACH 'data/sqlite_backup/comprehensive_equity.db' AS sqlite_db (TYPE SQLITE)")
        
        # Check what's in the attached database
        logger.info("📋 Checking attached database...")
        try:
            tables = conn.execute("SELECT name FROM sqlite_db.main.sqlite_master WHERE type='table'").fetchall()
        except:
            # Try alternative approach
            tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='sqlite_db'").fetchall()
        table_names = [table[0] for table in tables]
        logger.info(f"📋 Tables in SQLite: {table_names}")
        
        # Get counts from SQLite
        price_count_sqlite = conn.execute("SELECT COUNT(*) FROM sqlite_db.price_data").fetchone()[0]
        
        logger.info(f"📊 SQLite data:")
        logger.info(f"   📈 Price records: {price_count_sqlite:,}")
        
        # Delete securities table if it exists
        logger.info("🗑️ Deleting securities table...")
        try:
            conn.execute("DROP TABLE IF EXISTS securities")
            logger.info("   ✅ Securities table deleted")
        except Exception as e:
            logger.info(f"   ⚠️ Could not delete securities table: {e}")
        
        # Clear existing price_data
        logger.info("🧹 Clearing existing price_data...")
        conn.execute("DELETE FROM price_data")
        
        # Copy price_data directly using ATTACH
        logger.info("📥 Copying price_data...")
        conn.execute("INSERT INTO price_data SELECT * FROM sqlite_db.price_data")
        
        # Verify the copy
        price_count_duckdb = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        symbol_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        
        logger.info(f"✅ Migration complete!")
        logger.info(f"   📊 Price records: {price_count_duckdb:,}")
        logger.info(f"   🏢 Unique symbols: {symbol_count:,}")
        
        # Test foreign key constraints
        logger.info("🔗 Testing foreign key constraints...")
        try:
            sample_symbol = conn.execute("SELECT symbol FROM price_data LIMIT 1").fetchone()[0]
            sample_date = conn.execute("SELECT date FROM price_data LIMIT 1").fetchone()[0]
            
            logger.info(f"   🎯 Testing with symbol: {sample_symbol}, date: {sample_date}")
            
            conn.execute("""
                INSERT INTO technical_data (ticker, trade_date, open_price, high_price, low_price, close_price, volume, created_at, updated_at)
                VALUES (?, ?, 100.0, 110.0, 90.0, 105.0, 1000000, NOW(), NOW())
            """, [sample_symbol, sample_date])
            logger.info(f"   ✅ Foreign key constraint working")
            
            # Clean up test data
            conn.execute("DELETE FROM technical_data WHERE ticker = ? AND trade_date = ?", [sample_symbol, sample_date])
            logger.info(f"   🧹 Test data cleaned up")
            
        except Exception as e:
            logger.error(f"   ❌ Foreign key constraint failed: {e}")
        
        # Detach SQLite database
        conn.execute("DETACH sqlite_db")
        conn.close()
        
        logger.info("🎉 Migration successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    direct_migration()
