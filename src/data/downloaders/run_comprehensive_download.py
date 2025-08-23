#!/usr/bin/env python3
"""
Non-Interactive Comprehensive Equity Data Downloader
Runs automatically without user input for batch processing.
"""

from comprehensive_equity_data_downloader import ComprehensiveEquityDataDownloader
import sys

def main():
    """Run the comprehensive equity data downloader automatically."""
    print("🎯 NON-INTERACTIVE COMPREHENSIVE EQUITY DATA DOWNLOADER")
    print("=" * 70)
    
    # Initialize downloader
    downloader = ComprehensiveEquityDataDownloader()
    
    # Show current progress
    downloader.show_progress()
    
    # Get database stats
    stats = downloader.get_database_stats()
    if stats:
        print(f"\n📊 CURRENT DATABASE STATS:")
        print(f"Securities: {stats['securities_count']}")
        print(f"Active securities: {stats['active_securities']}")
        print(f"Total records: {stats['total_records']:,}")
        print(f"Unique symbols: {stats['unique_symbols']}")
        if stats['date_range'][0]:
            print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"Avg records per symbol: {stats['avg_records_per_symbol']:.1f}")
    
    # Get command line arguments for max companies
    max_companies = 10  # Default to 10 for testing
    if len(sys.argv) > 1:
        try:
            max_companies = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}. Using default: 10")
    
    print(f"\n🚀 Starting automatic download for {max_companies} companies...")
    print(f"This will download data from 2024-01-01 to today.")
    print(f"No user input required - running automatically...")
    
    # Start download automatically
    downloader.download_all_equity_data(max_companies=max_companies)
    
    # Show final stats
    print(f"\n📊 FINAL DATABASE STATS:")
    final_stats = downloader.get_database_stats()
    if final_stats:
        print(f"Securities: {final_stats['securities_count']}")
        print(f"Total records: {final_stats['total_records']:,}")
        print(f"Unique symbols: {final_stats['unique_symbols']}")
    
    print(f"\n🎉 COMPREHENSIVE EQUITY DATA DOWNLOAD COMPLETED!")
    print(f"📁 Database saved to: {downloader.db_path}")
    print(f"📄 Progress saved to: {downloader.progress_file}")

if __name__ == "__main__":
    main() 