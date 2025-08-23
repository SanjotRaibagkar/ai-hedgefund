#!/usr/bin/env python3
"""
Get ALL Indian Stocks
Fetches stocks from all available NSE indices to get 3000+ symbols.
"""

import sys
import os
from src.nsedata.NseUtility import NseUtils

def main():
    """Get all Indian stocks from all indices."""
    print("🚀 COMPREHENSIVE INDIAN STOCK COLLECTION")
    print("=" * 60)
    
    try:
        nse = NseUtils()
        print("✅ NSEUtility initialized")
        
        # Get all available indices
        all_indices = nse.equity_market_list
        print(f"📊 Total indices available: {len(all_indices)}")
        
        # Fetch stocks from all indices
        all_symbols = set()
        print("\n📥 Fetching stocks from all indices...")
        print("-" * 40)
        
        for idx in all_indices:
            try:
                symbols = nse.get_index_details(idx, list_only=True)
                if symbols:
                    all_symbols.update(symbols)
                    print(f"✅ {idx}: {len(symbols)} symbols")
                else:
                    print(f"⚠️ {idx}: No symbols found")
            except Exception as e:
                print(f"❌ {idx}: Error - {e}")
        
        print("=" * 60)
        print(f"🎯 TOTAL UNIQUE SYMBOLS: {len(all_symbols)}")
        print(f"📋 Sample symbols: {list(all_symbols)[:20]}")
        
        # Save to file
        symbols_list = sorted(list(all_symbols))
        with open("all_indian_stocks.txt", "w") as f:
            for symbol in symbols_list:
                f.write(f"{symbol}\n")
        
        print(f"💾 Saved {len(symbols_list)} symbols to all_indian_stocks.txt")
        print("🎉 Ready to download data for ALL Indian stocks!")
        
        return symbols_list
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    main() 