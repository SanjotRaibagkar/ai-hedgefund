#!/usr/bin/env python3
import sqlite3
import os
from src.nsedata.NseUtility import NseUtils
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

def main():
    print("🚀 FULL DATA DOWNLOAD STARTING")
    print("=" * 50)
    
    # Setup database
    db_path = 'data/full_indian_market.db'
    os.makedirs('data', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE IF NOT EXISTS price_data (symbol TEXT, date TEXT, close_price REAL, UNIQUE(symbol, date))')
    conn.commit()
    conn.close()
    
    # Load symbols
    with open('nse_equity_symbols_complete.txt', 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Loaded {len(symbols)} symbols")
    
    # Download data
    nse = NseUtils()
    successful = 0
    failed = 0
    start_time = time.time()
    
    def download_symbol(symbol):
        global successful, failed
        try:
            price_info = nse.price_info(symbol)
            if price_info:
                with sqlite3.connect(db_path) as conn:
                    conn.execute('INSERT OR REPLACE INTO price_data VALUES (?, ?, ?)', 
                               (symbol, datetime.now().strftime('%Y-%m-%d'), price_info.get('LastTradedPrice', 0)))
                    conn.commit()
                successful += 1
                return True
        except:
            failed += 1
            return False
    
    # Download first 50 symbols for testing
    test_symbols = symbols[:50]
    print(f"📥 Downloading data for {len(test_symbols)} symbols...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(download_symbol, test_symbols))
    
    total_time = time.time() - start_time
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Time: {total_time:.1f}s")
    print(f"📊 Rate: {successful/total_time:.1f} symbols/sec")
    
    # Test screening
    print("\n🧪 Testing screening...")
    with sqlite3.connect(db_path) as conn:
        symbols_with_data = conn.execute("SELECT DISTINCT symbol FROM price_data WHERE close_price > 0 LIMIT 10").fetchall()
    
    signals = []
    for (symbol,) in symbols_with_data:
        try:
            with sqlite3.connect(db_path) as conn:
                data = conn.execute("SELECT close_price FROM price_data WHERE symbol = ? ORDER BY date DESC LIMIT 5", (symbol,)).fetchall()
            
            if len(data) >= 3:
                current_price = data[0][0]
                avg_price = sum(row[0] for row in data) / len(data)
                
                if current_price > avg_price * 1.02:
                    signals.append(f"{symbol}: BULLISH @ ₹{current_price:.2f}")
                elif current_price < avg_price * 0.98:
                    signals.append(f"{symbol}: BEARISH @ ₹{current_price:.2f}")
        except:
            pass
    
    print(f"📋 Generated {len(signals)} signals:")
    for signal in signals[:5]:
        print(f"  • {signal}")
    
    print("\n🎉 TEST COMPLETED!")

if __name__ == "__main__":
    main() 