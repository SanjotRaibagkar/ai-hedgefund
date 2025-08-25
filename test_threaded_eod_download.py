#!/usr/bin/env python3
"""
Test Threaded EOD Extra Data Download
Demonstrates optimized threaded download capabilities and estimates download times.
"""

import time
import sys
from datetime import datetime, timedelta
from src.data.downloaders.eod_extra_data_downloader import EODExtraDataDownloader

def estimate_download_time():
    """Estimate download time for 5 years of data."""
    print("=" * 80)
    print("EOD EXTRA DATA DOWNLOAD TIME ESTIMATION")
    print("=" * 80)
    
    # Calculate trading days in 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    trading_days = 0
    current_dt = start_date
    while current_dt <= end_date:
        if current_dt.weekday() < 5:  # Monday to Friday
            trading_days += 1
        current_dt += timedelta(days=1)
    
    print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Total Trading Days: {trading_days:,}")
    print()
    
    # Estimate based on current performance
    print("⏱️  TIME ESTIMATES:")
    print("-" * 40)
    
    # Sequential download estimates (original method)
    avg_time_per_date = 2.5  # seconds per date (based on current performance)
    sequential_total_time = trading_days * avg_time_per_date * 3  # 3 data types
    sequential_hours = sequential_total_time / 3600
    
    print(f"🔄 Sequential Download (Original):")
    print(f"   • Estimated Time: {sequential_hours:.1f} hours")
    print(f"   • Time per Date: {avg_time_per_date:.1f} seconds")
    print(f"   • Total Time: {sequential_total_time/60:.1f} minutes")
    print()
    
    # Threaded download estimates (optimized method)
    threads = 10
    avg_time_per_date_threaded = 1.0  # seconds per date (reduced due to threading)
    threaded_total_time = (trading_days * avg_time_per_date_threaded * 3) / threads
    threaded_hours = threaded_total_time / 3600
    
    print(f"⚡ Threaded Download (Optimized):")
    print(f"   • Threads: {threads}")
    print(f"   • Estimated Time: {threaded_hours:.1f} hours")
    print(f"   • Time per Date: {avg_time_per_date_threaded:.1f} seconds")
    print(f"   • Total Time: {threaded_total_time/60:.1f} minutes")
    print(f"   • Speed Improvement: {sequential_total_time/threaded_total_time:.1f}x faster")
    print()
    
    # Data volume estimates
    print("📊 DATA VOLUME ESTIMATES:")
    print("-" * 40)
    
    # Based on current data sizes
    fno_records_per_day = 32000
    equity_records_per_day = 3000
    indices_records_per_day = 150
    
    total_fno_records = trading_days * fno_records_per_day
    total_equity_records = trading_days * equity_records_per_day
    total_indices_records = trading_days * indices_records_per_day
    total_records = total_fno_records + total_equity_records + total_indices_records
    
    print(f"📈 FNO Bhav Copy:")
    print(f"   • Records per Day: {fno_records_per_day:,}")
    print(f"   • Total Records: {total_fno_records:,}")
    print()
    
    print(f"📈 Equity Bhav Copy with Delivery:")
    print(f"   • Records per Day: {equity_records_per_day:,}")
    print(f"   • Total Records: {total_equity_records:,}")
    print()
    
    print(f"📈 Bhav Copy Indices:")
    print(f"   • Records per Day: {indices_records_per_day:,}")
    print(f"   • Total Records: {total_indices_records:,}")
    print()
    
    print(f"📊 TOTAL ESTIMATED DATA:")
    print(f"   • Total Records: {total_records:,}")
    print(f"   • Estimated Size: {total_records * 0.001:.1f} MB")
    print()
    
    return {
        'trading_days': trading_days,
        'sequential_hours': sequential_hours,
        'threaded_hours': threaded_hours,
        'speed_improvement': sequential_total_time/threaded_total_time,
        'total_records': total_records
    }

def test_threaded_download():
    """Test the threaded download functionality."""
    print("=" * 80)
    print("TESTING THREADED EOD DOWNLOAD")
    print("=" * 80)
    
    # Initialize downloader
    downloader = EODExtraDataDownloader()
    
    # Show current configuration
    print(f"🔧 Configuration:")
    print(f"   • Max Workers: {downloader.max_workers}")
    print(f"   • Delay between requests: {downloader.delay_between_requests}s")
    print(f"   • Retry attempts: {downloader.retry_attempts}")
    print()
    
    # Show current progress
    print("📈 Current Progress:")
    downloader.show_progress()
    
    # Get current database stats
    stats = downloader.get_database_stats()
    if stats:
        print("📊 Current Database Stats:")
        for data_type, data_stats in stats.items():
            print(f"   {data_type.upper()}:")
            print(f"     • Total Records: {data_stats['total_records']:,}")
            print(f"     • Unique Dates: {data_stats['unique_dates']}")
            if data_stats['earliest_date']:
                print(f"     • Date Range: {data_stats['earliest_date']} to {data_stats['latest_date']}")
        print()
    
    # Test small threaded download (last 7 days)
    print("🧪 Testing Threaded Download (Last 7 days)...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    start_time = time.time()
    
    # Test FNO threaded download
    print("📥 Testing FNO Bhav Copy (Threaded)...")
    fno_result = downloader.download_fno_bhav_copy_threaded(start_date, end_date)
    
    # Test Equity threaded download
    print("📥 Testing Equity Bhav Copy (Threaded)...")
    equity_result = downloader.download_equity_bhav_copy_delivery_threaded(start_date, end_date)
    
    # Test Indices threaded download
    print("📥 Testing Bhav Copy Indices (Threaded)...")
    indices_result = downloader.download_bhav_copy_indices_threaded(start_date, end_date)
    
    total_time = time.time() - start_time
    
    print("✅ Threaded Download Test Results:")
    print(f"   • FNO Records: {fno_result['total_records']:,}")
    print(f"   • Equity Records: {equity_result['total_records']:,}")
    print(f"   • Indices Records: {indices_result['total_records']:,}")
    print(f"   • Total Time: {total_time:.2f} seconds")
    print(f"   • Records per Second: {(fno_result['total_records'] + equity_result['total_records'] + indices_result['total_records']) / total_time:.2f}")
    print()

def main():
    """Main function."""
    print("🚀 EOD EXTRA DATA THREADED DOWNLOAD TEST")
    print("=" * 80)
    
    # Estimate download times
    estimates = estimate_download_time()
    
    # Test threaded download
    test_threaded_download()
    
    # Summary
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"📅 Trading Days (5 years): {estimates['trading_days']:,}")
    print(f"⏱️  Sequential Download: {estimates['sequential_hours']:.1f} hours")
    print(f"⚡ Threaded Download: {estimates['threaded_hours']:.1f} hours")
    print(f"🚀 Speed Improvement: {estimates['speed_improvement']:.1f}x faster")
    print(f"📊 Total Records: {estimates['total_records']:,}")
    print()
    
    print("💡 RECOMMENDATIONS:")
    print("   • Use threaded download for large datasets")
    print("   • Monitor system resources during download")
    print("   • Consider running during off-peak hours")
    print("   • Ensure stable internet connection")
    print()
    
    print("🎯 EXPECTED PERFORMANCE:")
    print(f"   • 5 Years Download: {estimates['threaded_hours']:.1f} hours")
    print(f"   • Records per Second: ~{estimates['total_records'] / (estimates['threaded_hours'] * 3600):.0f}")
    print(f"   • Speed vs Sequential: {estimates['speed_improvement']:.1f}x faster")
    print()

if __name__ == "__main__":
    main()
