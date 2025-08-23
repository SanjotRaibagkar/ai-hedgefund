import sqlite3
import duckdb
import pandas as pd
import os

def migrate_db(sqlite_file, duckdb_file):
    print(f"Migrating {sqlite_file} to {duckdb_file}")
    
    sqlite_conn = sqlite3.connect(sqlite_file)
    duckdb_conn = duckdb.connect(duckdb_file)
    
    tables = sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"  Table: {table_name}")
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
        if not df.empty:
            duckdb_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            duckdb_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            print(f"    Migrated {len(df)} records")
    
    sqlite_conn.close()
    duckdb_conn.close()
    print(f"✅ Done: {os.path.basename(duckdb_file)}")

# Migrate all databases
databases = [
    'data/comprehensive_equity.db',
    'data/final_comprehensive.db', 
    'data/full_indian_market.db',
    'data/indian_market.db',
    'data/optimized_market.db',
    'data/test_market.db'
]

for db in databases:
    if os.path.exists(db):
        duckdb_file = db.replace('.db', '.duckdb')
        migrate_db(db, duckdb_file)

print("🎉 All migrations completed!") 