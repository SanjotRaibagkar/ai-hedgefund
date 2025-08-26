#!/usr/bin/env python3
"""
Simple Intraday Data Collection Runner
Run data collection manually for testing.
"""

import sys
import os
sys.path.append('./src')

from loguru import logger
from intradayML import IntradayDataCollector

def setup_logging():
    """Setup logging."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
    )

def main():
    """Run data collection."""
    setup_logging()
    
    print("🚀 Intraday Data Collection")
    print("="*50)
    
    try:
        # Initialize data collector
        collector = IntradayDataCollector()
        
        # Collect data for both NIFTY and BANKNIFTY
        index_symbols = ['NIFTY', 'BANKNIFTY']
        
        for index_symbol in index_symbols:
            print(f"\n📊 Collecting data for {index_symbol}...")
            
            # Collect options chain data
            options_data = collector.collect_options_chain_data(index_symbol)
            print(f"   ✅ Options data: {len(options_data)} records")
            
            # Collect index data
            index_data = collector.collect_index_data(index_symbol)
            print(f"   ✅ Index data: {len(index_data)} records")
            
            # Collect labels data
            labels_data = collector.collect_labels_data(index_symbol)
            print(f"   ✅ Labels data: {len(labels_data)} records")
        
        # Collect FII/DII data
        print(f"\n💰 Collecting FII/DII data...")
        fii_dii_data = collector.collect_fii_dii_data()
        print(f"   ✅ FII/DII data: {len(fii_dii_data)} records")
        
        # Collect VIX data
        print(f"\n📊 Collecting VIX data...")
        vix_data = collector.collect_vix_data()
        print(f"   ✅ VIX data: {len(vix_data)} records")
        
        # Close connection
        collector.close()
        
        print("\n✅ Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in data collection: {e}")
        print(f"❌ Data collection failed: {e}")

if __name__ == "__main__":
    main()
