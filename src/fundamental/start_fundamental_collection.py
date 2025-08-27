#!/usr/bin/env python3
"""
Start Fundamental Data Collection
Simple script to start the fundamental data collection system.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.fundamental.collectors.nse_fundamental_collector import NSEFundamentalCollector
from src.fundamental.schedulers.fundamental_scheduler import FundamentalScheduler

def main():
    """Main function to start fundamental data collection."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("🚀 Starting Fundamental Data Collection System")
    print("=" * 60)
    print("📋 System Overview:")
    print("   Source: NSE Corporate Filings & Financial Results")
    print("   Data Types: Quarterly, Half-Yearly, Annual Reports")
    print("   Storage: DuckDB Database")
    print("   Features: Batch Processing, Progress Tracking, Error Handling")
    print("=" * 60)
    
    # Check if user wants to run scheduler or one-time download
    print("\n🎯 Choose operation mode:")
    print("   1. One-time download (download all fundamental data)")
    print("   2. Start scheduler (automated weekly/daily updates)")
    print("   3. Check status only")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # One-time download
            print("\n🚀 Starting one-time fundamental data download...")
            
            collector = NSEFundamentalCollector()
            
            # Get current status
            status = collector.get_download_status()
            print(f"\n📊 Current Status:")
            print(f"   Total Symbols: {status['total_symbols']}")
            print(f"   Completed: {status['completed_symbols']}")
            print(f"   Failed: {status['failed_symbols']}")
            print(f"   Progress: {status['progress_percentage']:.1f}%")
            
            # Ask for batch size
            try:
                batch_size = int(input("\nEnter batch size (default 50): ") or "50")
            except ValueError:
                batch_size = 50
            
            print(f"\n📦 Starting download with batch size: {batch_size}")
            print("⏰ This may take several hours depending on the number of symbols...")
            
            # Start download
            collector.download_all_fundamentals(batch_size=batch_size)
            
            # Final status
            final_status = collector.get_download_status()
            print(f"\n🎉 Download completed!")
            print(f"   Successfully processed: {final_status['completed_symbols']} symbols")
            print(f"   Failed: {final_status['failed_symbols']} symbols")
            print(f"   Final progress: {final_status['progress_percentage']:.1f}%")
            
        elif choice == "2":
            # Start scheduler
            print("\n🚀 Starting fundamental data scheduler...")
            print("📅 Schedule:")
            print("   Full download: Monday 6:00 AM")
            print("   Incremental update: Daily 8:00 PM")
            print("   Health check: Daily 10:00 AM")
            print("\n💡 Press Ctrl+C to stop the scheduler")
            
            scheduler = FundamentalScheduler()
            scheduler.run_scheduler()
            
        elif choice == "3":
            # Check status only
            print("\n📊 Checking fundamental data status...")
            
            collector = NSEFundamentalCollector()
            status = collector.get_download_status()
            
            print(f"\n📈 Fundamental Data Status:")
            print(f"   Total Symbols: {status['total_symbols']}")
            print(f"   Completed: {status['completed_symbols']}")
            print(f"   Failed: {status['failed_symbols']}")
            print(f"   Progress: {status['progress_percentage']:.1f}%")
            print(f"   Last Updated: {status['last_updated']}")
            
            # Show some sample data if available
            try:
                with collector.db_manager.connection as conn:
                    sample_data = conn.execute("""
                        SELECT symbol, period_type, report_date, revenue, net_profit 
                        FROM fundamental_data 
                        ORDER BY created_at DESC 
                        LIMIT 5
                    """).fetchdf()
                    
                    if not sample_data.empty:
                        print(f"\n📋 Recent Fundamental Data:")
                        print(sample_data.to_string(index=False))
                    else:
                        print(f"\n📋 No fundamental data found in database")
                        
            except Exception as e:
                print(f"⚠️ Could not retrieve sample data: {e}")
        
        else:
            print("❌ Invalid choice. Please run the script again and select 1, 2, or 3.")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Operation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
