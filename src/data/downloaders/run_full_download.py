#!/usr/bin/env python3
"""
Run Full Download
Downloads data for all NSE equity symbols using optimized downloader.
"""

from src.data.downloaders.optimized_equity_downloader import OptimizedEquityDataDownloader

def main():
    """Run the full download for all symbols."""
    print("🚀 STARTING FULL NSE EQUITY DATA DOWNLOAD (OPTIMIZED)")
    print("=" * 60)
    
    # Initialize downloader
    downloader = OptimizedEquityDataDownloader()
    
    try:
        # Start download for ALL symbols (no limit)
        print("\n🔄 Starting full download for ALL NSE equity symbols...")
        print("📅 Date range: 2024-01-01 to today")
        print("⏱️ Estimated time: 1-2 hours for all symbols (optimized)")
        print("💡 Progress will be saved automatically")
        print("🔄 System will resume from where it left off if interrupted")
        print()
        
        results = downloader.download_all_equity_data()  # No max_companies limit
        
        print("\n📊 FULL DOWNLOAD RESULTS:")
        print("=" * 40)
        print(f"✅ Completed Symbols: {results['completed']}")
        print(f"❌ Failed Symbols: {results['failed']}")
        print(f"📋 Total Symbols: {results['total_symbols']}")
        print(f"📈 Success Rate: {results['success_rate']:.2f}%")
        
        # Get database stats
        stats = downloader.get_database_stats()
        if stats:
            print(f"\n📊 FINAL DATABASE STATISTICS:")
            print(f"   • Total Securities: {stats['securities_count']}")
            print(f"   • FNO Securities: {stats['fno_securities_count']}")
            print(f"   • Price Records: {stats['price_data_count']:,}")
            print(f"   • Unique Symbols: {stats['unique_symbols']}")
            print(f"   • Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        
        print(f"\n📄 Files Created:")
        print(f"   • {downloader.db_path}")
        print(f"   • {downloader.progress_file}")
        print(f"   • optimized_equity_download.log")
        
        print(f"\n🎉 FULL DOWNLOAD COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("💡 You can resume the download by running this script again")

if __name__ == "__main__":
    main() 