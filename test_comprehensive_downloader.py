#!/usr/bin/env python3
"""
Test Comprehensive Equity Data Downloader
"""

from comprehensive_equity_data_downloader import ComprehensiveEquityDataDownloader

def main():
    """Test the comprehensive equity data downloader."""
    print("🎯 TESTING COMPREHENSIVE EQUITY DATA DOWNLOADER")
    print("=" * 60)
    
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
    
    # Start download (limit to 10 for testing)
    print(f"\n🚀 Starting download (limited to 10 companies for testing)...")
    downloader.download_all_equity_data(max_companies=10)
    
    # Show final stats
    print(f"\n📊 FINAL DATABASE STATS:")
    final_stats = downloader.get_database_stats()
    if final_stats:
        print(f"Securities: {final_stats['securities_count']}")
        print(f"Total records: {final_stats['total_records']:,}")
        print(f"Unique symbols: {final_stats['unique_symbols']}")
    
    print(f"\n🎉 TEST COMPLETED!")

if __name__ == "__main__":
    main() 