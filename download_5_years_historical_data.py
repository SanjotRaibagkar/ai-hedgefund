#!/usr/bin/env python3
"""
Download Historical EOD Data (2019-01-01 to Current)
Optimized threaded download for maximum performance.
"""

import time
import sys
from datetime import datetime
from src.data.downloaders.eod_extra_data_downloader import EODExtraDataDownloader

def main():
    """Download historical EOD data from 2019-01-01 to current date using threaded approach."""
    print("🚀 EOD EXTRA DATA - HISTORICAL DOWNLOAD (2019-01-01 to Current)")
    print("=" * 80)
    print("⚡ Using Optimized Threaded Download")
    print("🔧 Configuration: 10 threads, 0.2s delay between requests")
    print("📊 Expected: ~60M records, ~8-12 minutes download time")
    print("📅 Date Range: 2019-01-01 to Current Date")
    print("=" * 80)
    
    # Initialize downloader
    print("🔧 Initializing EOD Extra Data Downloader...")
    downloader = EODExtraDataDownloader(force_recreate_tables=False)
    
    # Show current progress
    print("\n📈 Current Progress:")
    downloader.show_progress()
    
    # Get current database stats
    stats = downloader.get_database_stats()
    if stats:
        print("\n📊 Current Database Stats:")
        for data_type, data_stats in stats.items():
            print(f"   {data_type.upper()}:")
            print(f"     • Total Records: {data_stats['total_records']:,}")
            print(f"     • Unique Dates: {data_stats['unique_dates']}")
            if data_stats['earliest_date']:
                print(f"     • Date Range: {data_stats['earliest_date']} to {data_stats['latest_date']}")
        print()
    
    # Confirm download
    print("⚠️  WARNING: This will download ~60M records and may take 8-12 minutes.")
    print("💾 Ensure you have sufficient disk space (~100MB).")
    print("🌐 Ensure stable internet connection.")
    print("💡 Existing data will be preserved (INSERT OR REPLACE)")
    
    response = input("\n❓ Proceed with historical download from 2019-01-01? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Download cancelled.")
        return
    
    # Start download
    print("\n🚀 Starting historical download from 2019-01-01...")
    print("⏱️  Estimated time: 8-12 minutes")
    print("📊 Expected records: ~60M")
    print("📅 Date Range: 2019-01-01 to Current Date")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Calculate years from 2019-01-01 to current date
        start_date = "2019-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        years_diff = (end_dt - start_dt).days / 365.25
        
        print(f"📅 Downloading data for {years_diff:.1f} years ({start_date} to {end_date})")
        
        # Download all data using threaded approach with custom date range
        results = downloader.download_all_eod_data_threaded(start_date, end_date)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Show results
        print("\n" + "=" * 80)
        print("✅ DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"📊 Total Records: {results['total_records']:,}")
        print(f"🚀 Records per Second: {results['records_per_second']:.2f}")
        if 'speed_improvement' in results:
            print(f"📈 Speed Improvement: {results['speed_improvement']:.1f}x faster than sequential")
        
        print("\n📋 DETAILED RESULTS:")
        for data_type, result in results['detailed_results'].items():
            print(f"   {data_type.upper()}:")
            print(f"     • Records: {result['total_records']:,}")
            print(f"     • Successful Dates: {result['successful_dates']}")
            print(f"     • Failed Dates: {result['failed_dates']}")
        
        # Show final database stats
        print("\n📊 FINAL DATABASE STATS:")
        final_stats = downloader.get_database_stats()
        if final_stats:
            for data_type, data_stats in final_stats.items():
                print(f"   {data_type.upper()}:")
                print(f"     • Total Records: {data_stats['total_records']:,}")
                print(f"     • Unique Dates: {data_stats['unique_dates']}")
                if data_stats['earliest_date']:
                    print(f"     • Date Range: {data_stats['earliest_date']} to {data_stats['latest_date']}")
        
        print("\n🎉 HISTORICAL DOWNLOAD COMPLETED!")
        print(f"📅 Date Range: {start_date} to {end_date}")
        print("📁 Data stored in: data/comprehensive_equity.duckdb")
        print("📈 Ready for analysis and backtesting!")
        
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user.")
        print("💡 Partial data has been saved. You can resume later.")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("💡 Check logs for details and try again.")

if __name__ == "__main__":
    main()
