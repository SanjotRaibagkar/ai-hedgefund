#!/usr/bin/env python3
"""
Check Download Progress
Shows current progress of the comprehensive download.
"""

import json
from comprehensive_equity_data_downloader import ComprehensiveEquityDataDownloader

def main():
    """Check and display current progress."""
    print("📊 COMPREHENSIVE DOWNLOAD PROGRESS")
    print("=" * 40)
    
    # Check progress file
    try:
        with open('comprehensive_progress.json', 'r') as f:
            progress = json.load(f)
        
        completed = len(progress['completed_symbols'])
        failed = len(progress['failed_symbols'])
        total = progress['total_symbols']
        
        print(f"📈 Progress Summary:")
        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📋 Total: {total}")
        print(f"   📊 Success Rate: {(completed/total*100):.1f}%")
        print(f"   ⏱️ Last Updated: {progress['last_updated']}")
        
    except FileNotFoundError:
        print("❌ Progress file not found")
        return
    
    # Check database stats
    try:
        downloader = ComprehensiveEquityDataDownloader()
        stats = downloader.get_database_stats()
        
        print(f"\n🗄️ Database Statistics:")
        print(f"   📊 Total Securities: {stats['securities_count']}")
        print(f"   🔥 FNO Securities: {stats['fno_securities_count']}")
        print(f"   📈 Price Records: {stats['price_data_count']:,}")
        print(f"   🎯 Unique Symbols: {stats['unique_symbols']}")
        print(f"   📅 Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")

if __name__ == "__main__":
    main()
