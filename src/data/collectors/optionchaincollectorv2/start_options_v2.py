#!/usr/bin/env python3
"""
Start Options Collection V2
Simple script to start the new two-thread options collection system.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.data.collectors.optionchaincollectorv2.options_threaded_manager import OptionsThreadedManager

def main():
    """Main function to start options collection."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("🚀 Starting Options Collection V2 System")
    print("=" * 50)
    print("📋 System Overview:")
    print("   Thread 1: Collect 1-minute options data → Parquet files")
    print("   Thread 2: Batch process parquet files → DuckDB every 5 minutes")
    print("   Benefits: Reduced database locks, better error handling")
    print("   Architecture: Threaded (no multiprocessing issues)")
    print("=" * 50)
    
    # Initialize threaded manager
    manager = OptionsThreadedManager()
    
    try:
        # Start all threads
        print("\n🚀 Starting threads...")
        results = manager.start_all_threads()
        
        if results['collection'] and results['batch']:
            print("✅ All threads started successfully!")
            
            # Show status
            status = manager.get_thread_status()
            print(f"\n📊 Thread Status:")
            print(f"   Collection Thread: {'🟢 Running' if status['collection_thread']['alive'] else '🔴 Stopped'}")
            print(f"   Batch Thread: {'🟢 Running' if status['batch_thread']['alive'] else '🔴 Stopped'}")
            
            print(f"\n📁 Data Storage:")
            print(f"   Parquet files: data/options_parquet/")
            print(f"   Processed files: data/options_processed/")
            print(f"   Database: options_chain_data.duckdb")
            
            print(f"\n⏰ Collection Schedule:")
            print(f"   Data collection: Every 1 minute")
            print(f"   Batch processing: Every 5 minutes")
            print(f"   Market hours: 9:15 AM - 3:30 PM (Mon-Fri)")
            
            print(f"\n💡 Monitoring:")
            print(f"   Check terminal for real-time logs")
            print(f"   Press Ctrl+C to stop all threads")
            
            # Run monitoring loop
            print(f"\n📊 Starting monitoring...")
            manager.run_monitoring_loop()
            
        else:
            print("❌ Failed to start all threads")
            print(f"   Collection: {'✅' if results['collection'] else '❌'}")
            print(f"   Batch: {'✅' if results['batch'] else '❌'}")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down...")
        manager.stop_all_threads()
        print("✅ All threads stopped")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        manager.stop_all_threads()


if __name__ == "__main__":
    main()
