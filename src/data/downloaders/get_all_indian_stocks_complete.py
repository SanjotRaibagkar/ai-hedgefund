#!/usr/bin/env python3
"""
Get ALL Indian Stocks - Complete List
Uses get_equity_full_list to get 3000+ Indian stocks.
"""

import sys
import os
from src.nsedata.NseUtility import NseUtils

def main():
    """Get all Indian stocks using get_equity_full_list."""
    print("🚀 GETTING ALL INDIAN STOCKS - COMPLETE LIST")
    print("=" * 60)
    
    try:
        nse = NseUtils()
        print("✅ NSEUtility initialized")
        
        # Method 1: Get equity full list
        print("\n📥 Method 1: Using get_equity_full_list...")
        try:
            all_stocks_full = nse.get_equity_full_list()
            print(f"✅ get_equity_full_list: {len(all_stocks_full)} stocks")
            print(f"📋 Sample: {list(all_stocks_full)[:10]}")
        except Exception as e:
            print(f"❌ get_equity_full_list error: {e}")
            all_stocks_full = []
        
        # Method 2: Get from all indices
        print("\n📥 Method 2: Using all indices...")
        all_indices = nse.equity_market_list
        all_symbols_indices = set()
        
        for idx in all_indices:
            try:
                symbols = nse.get_index_details(idx, list_only=True)
                if symbols:
                    all_symbols_indices.update(symbols)
            except Exception as e:
                continue
        
        print(f"✅ All indices: {len(all_symbols_indices)} unique symbols")
        
        # Method 3: Get F&O stocks
        print("\n📥 Method 3: Using F&O list...")
        try:
            fno_stocks = nse.get_fno_full_list()
            print(f"✅ F&O stocks: {len(fno_stocks)} stocks")
        except Exception as e:
            print(f"❌ F&O error: {e}")
            fno_stocks = []
        
        # Combine all methods
        all_symbols = set()
        if all_stocks_full:
            all_symbols.update(all_stocks_full)
        all_symbols.update(all_symbols_indices)
        if fno_stocks:
            all_symbols.update(fno_stocks)
        
        print("=" * 60)
        print(f"🎯 TOTAL UNIQUE SYMBOLS: {len(all_symbols)}")
        print(f"📋 Sample symbols: {list(all_symbols)[:20]}")
        
        # Save to file
        symbols_list = sorted(list(all_symbols))
        with open("all_indian_stocks_complete.txt", "w") as f:
            for symbol in symbols_list:
                f.write(f"{symbol}\n")
        
        print(f"💾 Saved {len(symbols_list)} symbols to all_indian_stocks_complete.txt")
        
        if len(symbols_list) >= 3000:
            print("🎉 SUCCESS! Got 3000+ Indian stocks!")
        else:
            print(f"⚠️ Got {len(symbols_list)} stocks, need to find more methods for 3000+")
        
        return symbols_list
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    main() 