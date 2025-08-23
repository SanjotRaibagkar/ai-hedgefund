#!/usr/bin/env python3
"""
Download NSE Equity List
Downloads the comprehensive equity securities list from NSE India website.
"""

import requests
import pandas as pd
import os
from datetime import datetime
from loguru import logger

def download_nse_equity_list():
    """Download the NSE Equity segment CSV file."""
    print("🚀 DOWNLOADING NSE EQUITY SECURITIES LIST")
    print("=" * 60)
    
    # NSE Equity segment CSV URL
    url = "https://www.nseindia.com/api/equity-derivatives"
    
    # Alternative URLs to try
    urls_to_try = [
        "https://www.nseindia.com/api/equity-derivatives",
        "https://www.nseindia.com/market-data/securities-available-for-trading",
        "https://www.nseindia.com/api/equity-stockIndices",
        "https://www.nseindia.com/api/equity-stockIndices?index=Securities%20in%20F%26O"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    print("📥 Attempting to download NSE Equity list...")
    
    for i, url in enumerate(urls_to_try):
        try:
            print(f"🔗 Trying URL {i+1}: {url}")
            
            # First get the main page to establish session
            session = requests.Session()
            main_page = session.get("https://www.nseindia.com", headers=headers)
            
            # Now try to get the equity data
            response = session.get(url, headers=headers)
            
            if response.status_code == 200:
                print(f"✅ Success! Status: {response.status_code}")
                print(f"📊 Content length: {len(response.content)} bytes")
                
                # Try to parse as CSV
                try:
                    # Save raw content first
                    with open("nse_equity_raw.csv", "wb") as f:
                        f.write(response.content)
                    print("💾 Saved raw content to nse_equity_raw.csv")
                    
                    # Try to parse as CSV
                    df = pd.read_csv("nse_equity_raw.csv")
                    print(f"📈 Parsed CSV with {len(df)} rows and {len(df.columns)} columns")
                    print(f"📋 Columns: {list(df.columns)}")
                    print(f"📊 Sample data:")
                    print(df.head())
                    
                    return df
                    
                except Exception as e:
                    print(f"⚠️ CSV parsing failed: {e}")
                    print("📄 Raw content preview:")
                    print(response.text[:500])
                    
                    # Try to extract symbols from text
                    lines = response.text.split('\n')
                    symbols = []
                    for line in lines:
                        if ',' in line and len(line.strip()) > 0:
                            parts = line.split(',')
                            if len(parts) > 0 and parts[0].strip().isalpha():
                                symbols.append(parts[0].strip())
                    
                    print(f"🎯 Extracted {len(symbols)} potential symbols")
                    return symbols
                    
            else:
                print(f"❌ Failed with status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error with URL {i+1}: {e}")
            continue
    
    print("❌ All URLs failed. Trying alternative approach...")
    return None

def get_equity_list_from_nse_website():
    """Alternative approach to get equity list."""
    print("\n🔄 ALTERNATIVE APPROACH: Using NSEUtility with enhanced methods")
    print("=" * 60)
    
    try:
        from src.nsedata.NseUtility import NseUtils
        nse = NseUtils()
        
        all_symbols = set()
        
        # Method 1: Get from all working indices
        print("📥 Method 1: Getting from all working indices...")
        working_indices = [
            'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY MIDCAP 50', 'NIFTY MIDCAP 100',
            'NIFTY SMALLCAP 50', 'NIFTY SMALLCAP 100', 'NIFTY SMALLCAP 250',
            'NIFTY BANK', 'NIFTY IT', 'NIFTY PHARMA', 'NIFTY AUTO', 'NIFTY FMCG',
            'NIFTY METAL', 'NIFTY REALTY', 'Securities in F&O'
        ]
        
        for idx in working_indices:
            try:
                symbols = nse.get_index_details(idx, list_only=True)
                if symbols:
                    all_symbols.update(symbols)
                    print(f"✅ {idx}: {len(symbols)} symbols")
            except Exception as e:
                print(f"⚠️ {idx}: Error - {e}")
        
        # Method 2: Try to get F&O list
        print("\n📥 Method 2: Getting F&O list...")
        try:
            fno_stocks = nse.get_fno_full_list()
            if fno_stocks:
                all_symbols.update(fno_stocks)
                print(f"✅ F&O: {len(fno_stocks)} symbols")
        except Exception as e:
            print(f"⚠️ F&O error: {e}")
        
        # Method 3: Try most active stocks
        print("\n📥 Method 3: Getting most active stocks...")
        try:
            active_stocks = nse.most_active_equity_stocks_by_volume()
            if active_stocks:
                symbols = [stock[0] for stock in active_stocks if stock[0]]
                all_symbols.update(symbols)
                print(f"✅ Active stocks: {len(symbols)} symbols")
        except Exception as e:
            print(f"⚠️ Active stocks error: {e}")
        
        symbols_list = sorted(list(all_symbols))
        print(f"\n🎯 TOTAL UNIQUE SYMBOLS: {len(symbols_list)}")
        print(f"📋 Sample symbols: {symbols_list[:15]}")
        
        # Save to file
        with open("nse_equity_symbols.txt", "w") as f:
            for symbol in symbols_list:
                f.write(f"{symbol}\n")
        
        print(f"💾 Saved {len(symbols_list)} symbols to nse_equity_symbols.txt")
        
        return symbols_list
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    """Main function to get NSE equity list."""
    print("🎯 NSE EQUITY SECURITIES DOWNLOAD")
    print("=" * 60)
    
    # Try direct download first
    result = download_nse_equity_list()
    
    if result is None or len(result) < 1000:
        print("\n🔄 Falling back to NSEUtility method...")
        result = get_equity_list_from_nse_website()
    
    if result:
        print(f"\n🎉 SUCCESS! Got {len(result)} equity symbols")
        print("📊 Ready for comprehensive data download and screening!")
    else:
        print("\n❌ Failed to get equity list")

if __name__ == "__main__":
    main() 