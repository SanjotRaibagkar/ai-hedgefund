#!/usr/bin/env python3
"""Migrate SQLite to DuckDB"""

import sqlite3
import duckdb
import os
import pandas as pd

def migrate_database(sqlite_path, duckdb_path):
    """Migrate SQLite database to DuckDB."""
    print(f"Migrating {os.path.basename(sqlite_path)} to DuckDB...")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    
    # Connect to DuckDB
    duckdb_conn = duckdb.connect(duckdb_path)
    
    # Get tables
    tables = sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"  Migrating table: {table_name}")
        
        # Read from SQLite
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
        
        if not df.empty:
            # Write to DuckDB
            duckdb_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            duckdb_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            print(f"    Migrated {len(df)} records")
    
    sqlite_conn.close()
    duckdb_conn.close()
    print(f"✅ Migration completed: {os.path.basename(duckdb_path)}")

def main():
    """Migrate all SQLite databases."""
    data_dir = "data"
    
    # Find SQLite databases
    sqlite_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.db') or file.endswith('.sqlite'):
                sqlite_files.append(os.path.join(data_dir, file))
    
    print(f"Found {len(sqlite_files)} SQLite databases:")
    for file in sqlite_files:
        print(f"  • {os.path.basename(file)}")
    
    # Migrate each database
    for sqlite_file in sqlite_files:
        duckdb_file = sqlite_file.replace('.db', '.duckdb').replace('.sqlite', '.duckdb')
        migrate_database(sqlite_file, duckdb_file)

if __name__ == "__main__":
    main() 