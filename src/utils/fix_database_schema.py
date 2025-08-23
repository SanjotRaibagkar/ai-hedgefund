#!/usr/bin/env python3
"""
Fix Database Schema
Adds missing columns to existing database.
"""

import sqlite3
import os

def fix_database_schema():
    """Fix the database schema by adding missing columns."""
    db_path = "data/comprehensive_equity.db"
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist. Creating new one...")
        return
    
    print(f"Fixing database schema for {db_path}...")
    
    conn = sqlite3.connect(db_path)
    
    # Check existing columns
    cursor = conn.execute("PRAGMA table_info(securities)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Add missing columns
    missing_columns = [
        ('is_fno', 'BOOLEAN DEFAULT 0'),
        ('series', 'TEXT'),
        ('listing_date', 'TEXT'),
        ('face_value', 'REAL'),
        ('company_name', 'TEXT'),
        ('is_active', 'BOOLEAN DEFAULT 1'),
        ('last_updated', 'TEXT'),
        ('data_start_date', 'TEXT'),
        ('data_end_date', 'TEXT'),
        ('total_records', 'INTEGER DEFAULT 0')
    ]
    
    for column_name, column_type in missing_columns:
        if column_name not in columns:
            print(f"Adding {column_name} column to securities table...")
            try:
                conn.execute(f'ALTER TABLE securities ADD COLUMN {column_name} {column_type}')
                print(f"✅ {column_name} column added successfully")
            except sqlite3.OperationalError as e:
                print(f"⚠️ Error adding {column_name}: {e}")
        else:
            print(f"✅ {column_name} column already exists")
    
    conn.commit()
    conn.close()
    
    print("Database schema fixed successfully!")

if __name__ == "__main__":
    fix_database_schema()
