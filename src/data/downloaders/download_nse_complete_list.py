#!/usr/bin/env python3
"""
Download Complete NSE Equity List
Downloads the comprehensive CSV file with all active securities from NSE India.
"""

import requests
import pandas as pd
import os
from datetime import datetime

def download_nse_complete_list():
    """Download the complete NSE equity list CSV file."""
    print("🚀 DOWNLOADING COMPLETE NSE EQUITY LIST")
    print("=" * 60)
    
    # NSE Equity CSV URL (found from the website)
    url = "https://nsearchives.nseindia.com/content/equities/List_of_Active_Securities_CM_DEBT.csv"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }
    
    try:
        print(f"📥 Downloading from: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Success! Status: {response.status_code}")
            print(f"📊 Content length: {len(response.content)} bytes")
            
            # Save the CSV file
            filename = "nse_complete_equity_list.csv"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"💾 Saved to: {filename}")
            
            # Parse the CSV
            try:
                df = pd.read_csv(filename)
                print(f"📈 CSV loaded successfully!")
                print(f"📊 Total rows: {len(df)}")
                print(f"📋 Total columns: {len(df.columns)}")
                print(f"📋 Column names: {list(df.columns)}")
                
                print(f"\n📄 Sample data:")
                print(df.head())
                
                # Extract equity symbols (assuming first column contains symbols)
                if len(df.columns) > 0:
                    symbol_column = df.columns[0]
                    symbols = df[symbol_column].dropna().tolist()
                    
                    # Filter out non-symbol entries
                    valid_symbols = [str(sym).strip() for sym in symbols if str(sym).strip() and len(str(sym).strip()) > 0 and not str(sym).strip().isdigit()]
                    
                    print(f"\n🎯 ANALYSIS:")
                    print(f"📊 Total entries: {len(symbols)}")
                    print(f"✅ Valid symbols: {len(valid_symbols)}")
                    print(f"📋 Sample symbols: {valid_symbols[:20]}")
                    
                    # Save symbols to text file
                    symbols_file = "nse_complete_symbols.txt"
                    with open(symbols_file, "w") as f:
                        for symbol in valid_symbols:
                            f.write(f"{symbol}\n")
                    
                    print(f"💾 Saved {len(valid_symbols)} symbols to: {symbols_file}")
                    
                    if len(valid_symbols) >= 3000:
                        print(f"🎉 SUCCESS! Got {len(valid_symbols)} Indian stocks (3000+ target achieved!)")
                    else:
                        print(f"⚠️ Got {len(valid_symbols)} symbols, need to find more for 3000+ target")
                    
                    return valid_symbols
                else:
                    print("❌ No columns found in CSV")
                    return []
                    
            except Exception as e:
                print(f"❌ Error parsing CSV: {e}")
                print("📄 Raw content preview:")
                print(response.text[:500])
                return []
                
        else:
            print(f"❌ Failed with status: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return []

def main():
    """Main function."""
    print("🎯 NSE COMPLETE EQUITY LIST DOWNLOAD")
    print("=" * 60)
    
    symbols = download_nse_complete_list()
    
    if symbols:
        print(f"\n🎉 DOWNLOAD COMPLETED!")
        print(f"📊 Total symbols: {len(symbols)}")
        print("📈 Ready for comprehensive data download and screening!")
        print("🚀 System can now handle 3000+ Indian stocks!")
    else:
        print("\n❌ Download failed")

if __name__ == "__main__":
    main() 