#!/usr/bin/env python3
"""
Full Data Download and Screening Test
"""

import sqlite3
import pandas as pd
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from src.nsedata.NseUtility import NseUtils

def init_database():
    """Initialize database."""
    db_path = "data/full_indian_market.db"
    os.makedirs("data", exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                last_updated TEXT,
                UNIQUE(symbol, date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)")
        conn.commit()
    return db_path

def load_symbols():
    """Load all symbols."""
    symbols = set()
    
    # Load from CSV
    if os.path.exists('nse_equity_symbols_complete.txt'):
        with open('nse_equity_symbols_complete.txt', 'r') as f:
            symbols.update([line.strip() for line in f if line.strip()])
    
    # Load from indices
    nse = NseUtils()
    indices = ['NIFTY 50', 'NIFTY NEXT 50', 'NIFTY BANK']
    for idx in indices:
        try:
            index_symbols = nse.get_index_details(idx, list_only=True)
            if index_symbols:
                symbols.update(index_symbols)
        except:
            pass
    
    return sorted(list(symbols))

def download_symbol_data(symbol):
    """Download data for one symbol."""
    try:
        nse = NseUtils()
        price_info = nse.price_info(symbol)
        if price_info:
            return {
                'symbol': symbol,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'open_price': price_info.get('Open', 0),
                'high_price': price_info.get('High', 0),
                'low_price': price_info.get('Low', 0),
                'close_price': price_info.get('LastTradedPrice', 0),
                'volume': price_info.get('Volume', 0),
                'last_updated': datetime.now().isoformat()
            }
    except:
        pass
    return None

def store_data(db_path, data_list):
    """Store data in database."""
    with sqlite3.connect(db_path) as conn:
        for data in data_list:
            if data:
                conn.execute("""
                    INSERT OR REPLACE INTO price_data 
                    (symbol, date, open_price, high_price, low_price, close_price, volume, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['symbol'], data['date'], data['open_price'], data['high_price'],
                    data['low_price'], data['close_price'], data['volume'], data['last_updated']
                ))
        conn.commit()

def download_all_data(symbols, max_symbols=50):
    """Download data for all symbols."""
    print(f"🚀 Downloading data for {min(len(symbols), max_symbols)} symbols...")
    
    db_path = init_database()
    symbols_to_download = symbols[:max_symbols]
    
    start_time = time.time()
    successful = 0
    failed = 0
    data_list = []
    
    # Process in batches
    batch_size = 10
    for i in range(0, len(symbols_to_download), batch_size):
        batch = symbols_to_download[i:i+batch_size]
        print(f"📥 Batch {i//batch_size + 1}: {len(batch)} symbols")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            batch_data = list(executor.map(download_symbol_data, batch))
        
        # Count results
        for data in batch_data:
            if data:
                data_list.append(data)
                successful += 1
            else:
                failed += 1
        
        # Store batch
        if data_list:
            store_data(db_path, data_list)
            data_list = []
    
    total_time = time.time() - start_time
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Time: {total_time:.1f}s")
    print(f"📊 Rate: {successful/total_time:.1f} symbols/sec")
    
    return db_path

def test_screening(db_path):
    """Test screening system."""
    print("\n🧪 Testing screening system...")
    
    with sqlite3.connect(db_path) as conn:
        symbols_with_data = conn.execute("""
            SELECT DISTINCT symbol FROM price_data 
            WHERE close_price > 0 
            LIMIT 20
        """).fetchall()
    
    symbols = [row[0] for row in symbols_with_data]
    signals = []
    
    for symbol in symbols:
        try:
            with sqlite3.connect(db_path) as conn:
                data = pd.read_sql_query("""
                    SELECT * FROM price_data 
                    WHERE symbol = ? 
                    ORDER BY date DESC 
                    LIMIT 10
                """, conn, params=(symbol,))
            
            if len(data) >= 5:
                current_price = data.iloc[0]['close_price']
                avg_price = data['close_price'].mean()
                
                if current_price > avg_price * 1.02:
                    signals.append({
                        'symbol': symbol,
                        'price': current_price,
                        'signal': 'BULLISH',
                        'entry': current_price,
                        'sl': current_price * 0.95,
                        'target': current_price * 1.10
                    })
                elif current_price < avg_price * 0.98:
                    signals.append({
                        'symbol': symbol,
                        'price': current_price,
                        'signal': 'BEARISH',
                        'entry': current_price,
                        'sl': current_price * 1.05,
                        'target': current_price * 0.90
                    })
        except:
            pass
    
    # Save results
    if signals:
        df = pd.DataFrame(signals)
        df.to_csv('screening_results.csv', index=False)
        print(f"📄 Saved {len(signals)} signals to screening_results.csv")
        
        print(f"\n📋 Sample signals:")
        for signal in signals[:5]:
            print(f"  • {signal['symbol']}: {signal['signal']} @ ₹{signal['entry']:.2f}")
    else:
        print("⚠️ No signals generated")

def main():
    """Main function."""
    print("🎯 FULL DATA DOWNLOAD AND SCREENING TEST")
    print("=" * 50)
    
    # Load symbols
    symbols = load_symbols()
    print(f"📊 Loaded {len(symbols)} symbols")
    
    # Download data
    db_path = download_all_data(symbols, max_symbols=50)
    
    # Test screening
    test_screening(db_path)
    
    print(f"\n🎉 TEST COMPLETED!")

if __name__ == "__main__":
    main() 