#!/usr/bin/env python3
"""Show NSE symbols summary."""

import json
import pandas as pd

def main():
    # Load the symbols data
    with open('all_nse_symbols.json', 'r') as f:
        data = json.load(f)
    
    print("📊 NSE SYMBOLS COLLECTION SUMMARY")
    print("=" * 50)
    print(f"✅ Total Unique Symbols: {data['total_symbols']}")
    print(f"🕐 Collection Time: {data['collection_time']}")
    print()
    
    print("📋 Methods Used:")
    for method, symbols in data['methods_used'].items():
        print(f"   • {method}: {len(symbols)} symbols")
    print()
    
    print("📋 Sample Symbols (first 30):")
    for symbol in data['all_symbols'][:30]:
        print(f"   • {symbol}")
    
    if len(data['all_symbols']) > 30:
        print(f"   ... and {len(data['all_symbols']) - 30} more")
    
    print()
    print("📄 Files Created:")
    print("   • all_nse_symbols.json")
    print("   • all_nse_symbols.csv")
    print("   • nse_equity_latest.csv")

if __name__ == "__main__":
    main()
