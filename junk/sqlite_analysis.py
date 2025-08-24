#!/usr/bin/env python3
"""
SQLite Usage Analysis
Analyze all SQLite usage in the codebase to identify what needs to be migrated to DuckDB.
"""

import os
import re
from pathlib import Path

def analyze_sqlite_usage():
    """Analyze all SQLite usage in the codebase."""
    print("🔍 SQLITE USAGE ANALYSIS")
    print("=" * 60)
    
    # Define categories
    categories = {
        'core_components': [],
        'data_managers': [],
        'providers': [],
        'utils': [],
        'tests': [],
        'docs': [],
        'junk': [],
        'migration_scripts': []
    }
    
    # Files that use SQLite
    sqlite_files = [
        # Core Components
        ('src/data/database/duckdb_manager.py', 'core_components', 'DatabaseManager class still uses SQLite'),
        ('src/data/providers/database_provider.py', 'providers', 'DatabaseProvider supports both SQLite and DuckDB'),
        
        # Data Managers
        ('src/data/indian_market_data_manager.py', 'data_managers', 'IndianMarketDataManager uses SQLite'),
        ('src/data/indian_data_manager.py', 'data_managers', 'IndianDataManager uses SQLite'),
        ('src/data/enhanced_indian_data_manager.py', 'data_managers', 'EnhancedIndianDataManager uses SQLite'),
        
        # Utils
        ('src/utils/migrate_to_duckdb.py', 'utils', 'Migration utility'),
        ('src/utils/migrate.py', 'utils', 'Migration utility'),
        ('src/utils/fix_database_schema.py', 'utils', 'Schema fix utility'),
        ('src/utils/database_stats_utility.py', 'utils', 'Database stats utility'),
        ('src/utils/check_database_stats.py', 'utils', 'Database stats checker'),
        
        # Tests
        ('tests/test_enhanced_eod_system.py', 'tests', 'EOD system test'),
        ('tests/test_ui_with_comprehensive_data.py', 'tests', 'UI test with comprehensive data'),
        ('tests/test_screening_with_comprehensive_data.py', 'tests', 'Screening test'),
        ('tests/test_screening_simple.py', 'tests', 'Simple screening test'),
        ('tests/test_screening_large_sample.py', 'tests', 'Large sample screening test'),
        ('tests/test_screening_database_only.py', 'tests', 'Database-only screening test'),
        ('tests/test_download.py', 'tests', 'Download test'),
        ('tests/optimized_test.py', 'tests', 'Optimized test'),
        ('tests/final_comprehensive_test.py', 'tests', 'Final comprehensive test'),
        
        # Junk files (migration scripts)
        ('junk/working_migration.py', 'junk', 'Working migration script'),
        ('junk/fix_migration.py', 'junk', 'Fix migration script'),
        ('junk/final_migration.py', 'junk', 'Final migration script'),
        ('junk/simple_migrate.py', 'junk', 'Simple migration script'),
        ('junk/simple_fix.py', 'junk', 'Simple fix script'),
        ('junk/migrate_comprehensive_to_duckdb.py', 'junk', 'Comprehensive migration script'),
        ('junk/attach_migration.py', 'junk', 'ATTACH migration script'),
        
        # Downloaders
        ('src/data/downloaders/download_test.py', 'utils', 'Download test utility'),
        ('src/data/downloaders/full_data_download.py', 'utils', 'Full data download utility'),
    ]
    
    # Categorize files
    for file_path, category, description in sqlite_files:
        if os.path.exists(file_path):
            categories[category].append((file_path, description))
    
    # Print analysis
    total_files = 0
    for category, files in categories.items():
        if files:
            print(f"\n📁 {category.upper().replace('_', ' ')} ({len(files)} files):")
            print("-" * 50)
            for file_path, description in files:
                print(f"  • {file_path}")
                print(f"    {description}")
                total_files += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total files using SQLite: {total_files}")
    print(f"  Core components: {len(categories['core_components'])}")
    print(f"  Data managers: {len(categories['data_managers'])}")
    print(f"  Providers: {len(categories['providers'])}")
    print(f"  Utils: {len(categories['utils'])}")
    print(f"  Tests: {len(categories['tests'])}")
    print(f"  Junk/Migration: {len(categories['junk'])}")
    
    # Priority analysis
    print(f"\n🎯 MIGRATION PRIORITY:")
    print("=" * 40)
    
    print(f"\n🔥 HIGH PRIORITY (Core Components):")
    for file_path, description in categories['core_components']:
        print(f"  • {file_path} - {description}")
    
    print(f"\n⚡ MEDIUM PRIORITY (Data Managers & Providers):")
    for file_path, description in categories['data_managers'] + categories['providers']:
        print(f"  • {file_path} - {description}")
    
    print(f"\n🔧 LOW PRIORITY (Utils & Tests):")
    for file_path, description in categories['utils'] + categories['tests']:
        print(f"  • {file_path} - {description}")
    
    print(f"\n🗑️  NO ACTION NEEDED (Junk/Migration):")
    for file_path, description in categories['junk']:
        print(f"  • {file_path} - {description}")

def check_database_files():
    """Check what database files exist."""
    print(f"\n🗄️ DATABASE FILES ANALYSIS:")
    print("=" * 40)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("  ❌ data/ directory not found")
        return
    
    db_files = []
    for file in data_dir.iterdir():
        if file.is_file() and (file.suffix in ['.db', '.duckdb', '.sqlite']):
            size_mb = file.stat().st_size / (1024 * 1024)
            db_files.append((file.name, size_mb))
    
    if db_files:
        print(f"  📁 Found {len(db_files)} database files:")
        for name, size in sorted(db_files, key=lambda x: x[1], reverse=True):
            print(f"    • {name} ({size:.1f} MB)")
    else:
        print("  ❌ No database files found")

def main():
    """Main analysis function."""
    analyze_sqlite_usage()
    check_database_files()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 40)
    print("1. 🔥 Migrate core components first (DatabaseManager, DatabaseProvider)")
    print("2. ⚡ Update data managers to use DuckDB")
    print("3. 🔧 Update utils and tests as needed")
    print("4. 🗑️  Keep junk/migration files as-is (they're temporary)")
    print("5. 📚 Update documentation to reflect DuckDB usage")

if __name__ == "__main__":
    main()
