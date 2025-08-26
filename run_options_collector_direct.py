#!/usr/bin/env python3
"""
Direct Options Chain Collector Runner
Runs the options chain collector directly without encoding issues.
"""

import sys
import os
import time
import schedule
from datetime import datetime

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data.downloaders.options_chain_collector import OptionsChainCollector

def main():
    """Main function to run the options chain collector."""
    print("Starting Options Chain Collector...")
    
    try:
        # Initialize the collector
        collector = OptionsChainCollector()
        
        # Run initial collection
        print("Running initial data collection...")
        collector.collect_all_data()
        
        # Schedule regular collection every 3 minutes
        print("Scheduling data collection every 3 minutes...")
        schedule.every(3).minutes.do(collector.collect_all_data)
        
        # Main loop
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("Stopping Options Chain Collector...")
    except Exception as e:
        print(f"Error in Options Chain Collector: {e}")

if __name__ == "__main__":
    main()
