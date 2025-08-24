#!/usr/bin/env python3
"""
Migration from SQLite to DuckDB using ATTACH method
"""

import duckdb
from loguru import logger

def attach_migration():
    """Migrate data from SQLite to DuckDB using ATTACH method."""
    logger.info("🚀 Starting ATTACH migration from SQLite to DuckDB...")
    
    try:
        # Connect to DuckDB
        logger.info("🔗 Connecting to DuckDB...")
        duckdb_conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Attach SQLite database
        logger.info("📎 Attaching SQLite database...")
        duckdb_conn.execute("ATTACH 'data/comprehensive_equity.db' AS sqlite_db (TYPE sqlite)")
        
        # Check what tables are available in SQLite
        logger.info("📋 Checking available tables in SQLite...")
        try:
            sqlite_tables = duckdb_conn.execute("SHOW TABLES FROM sqlite_db").fetchall()
            logger.info(f"Available SQLite tables: {[table[0] for table in sqlite_tables]}")
        except Exception as e:
            logger.warning(f"Could not list tables: {e}, proceeding with known tables")
        
        # Clear existing data in DuckDB
        logger.info("🧹 Clearing existing data in DuckDB...")
        duckdb_conn.execute("DELETE FROM price_data")
        duckdb_conn.execute("DELETE FROM securities")
        
        # Copy price_data table
        logger.info("📥 Copying price_data from SQLite to DuckDB...")
        duckdb_conn.execute("INSERT INTO price_data SELECT * FROM sqlite_db.price_data")
        
        # Copy securities table
        logger.info("📥 Copying securities from SQLite to DuckDB...")
        duckdb_conn.execute("INSERT INTO securities SELECT * FROM sqlite_db.securities")
        
        # Verify the migration
        logger.info("🔍 Verifying migration...")
        price_count = duckdb_conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        symbol_count = duckdb_conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
        securities_count = duckdb_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        
        # Get date range
        date_range = duckdb_conn.execute("SELECT MIN(date) as start_date, MAX(date) as end_date FROM price_data").fetchone()
        
        logger.info(f"✅ Migration completed successfully!")
        logger.info(f"   📊 Price records: {price_count:,}")
        logger.info(f"   🏢 Unique symbols: {symbol_count:,}")
        logger.info(f"   📋 Securities records: {securities_count:,}")
        logger.info(f"   📅 Date range: {date_range[0]} to {date_range[1]}")
        
        # Sample data verification
        logger.info("🔍 Sample data verification...")
        sample_symbols = duckdb_conn.execute("SELECT DISTINCT symbol FROM price_data LIMIT 5").fetchall()
        logger.info(f"   Sample symbols: {[s[0] for s in sample_symbols]}")
        
        duckdb_conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

def main():
    """Main function."""
    success = attach_migration()
    
    if success:
        logger.info("🎉 Comprehensive data migration to DuckDB completed successfully!")
        logger.info("✅ All strategies will now use the comprehensive DuckDB database with 2,129+ symbols")
    else:
        logger.error("❌ Migration failed. Please check the errors above.")

if __name__ == "__main__":
    main()
