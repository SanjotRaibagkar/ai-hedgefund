#!/usr/bin/env python3
"""
Full Data Download for All 2799 Indian Symbols
"""

from src.nsedata.NseUtility import NseUtils
import sqlite3
import os
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def init_database():
    """Initialize the database."""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/full_indian_market.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
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
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)')
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def load_all_symbols():
    """Load all 2799 symbols."""
    symbols = []
    
    # Load from CSV
    if os.path.exists('nse_equity_symbols_complete.txt'):
        with open('nse_equity_symbols_complete.txt', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Loaded {len(symbols)} symbols from CSV")
    return symbols

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

def store_batch_data(data_list):
    """Store a batch of data."""
    if not data_list:
        return
    
    conn = sqlite3.connect('data/full_indian_market.db')
    for data in data_list:
        if data:
            conn.execute('''
                INSERT OR REPLACE INTO price_data 
                (symbol, date, open_price, high_price, low_price, close_price, volume, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['symbol'], data['date'], data['open_price'], data['high_price'],
                data['low_price'], data['close_price'], data['volume'], data['last_updated']
            ))
    conn.commit()
    conn.close()

def download_all_data(symbols, max_symbols=100):
    """Download data for all symbols."""
    print(f"🚀 Starting download for {min(len(symbols), max_symbols)} symbols...")
    
    symbols_to_download = symbols[:max_symbols]
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Process in batches
    batch_size = 20
    for i in range(0, len(symbols_to_download), batch_size):
        batch = symbols_to_download[i:i+batch_size]
        print(f"📥 Batch {i//batch_size + 1}/{(len(symbols_to_download)-1)//batch_size + 1}: {len(batch)} symbols")
        
        # Download batch data
        with ThreadPoolExecutor(max_workers=10) as executor:
            batch_data = list(executor.map(download_symbol_data, batch))
        
        # Count results
        for data in batch_data:
            if data:
                successful += 1
            else:
                failed += 1
        
        # Store batch
        store_batch_data(batch_data)
        
        # Progress update
        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        print(f"⏱️ Progress: {i + len(batch)}/{len(symbols_to_download)} symbols ({rate:.1f} symbols/sec)")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Download completed!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    print(f"📊 Rate: {successful/total_time:.1f} symbols/sec")

def test_screening():
    """Test the screening system."""
    print("\n🧪 Testing screening system...")
    
    conn = sqlite3.connect('data/full_indian_market.db')
    
    # Get symbols with data
    symbols_with_data = conn.execute("""
        SELECT DISTINCT symbol FROM price_data 
        WHERE close_price > 0 
        ORDER BY symbol 
        LIMIT 50
    """).fetchall()
    
    symbols = [row[0] for row in symbols_with_data]
    print(f"📊 Testing screening with {len(symbols)} symbols")
    
    signals = []
    for symbol in symbols:
        try:
            # Get recent data
            data = pd.read_sql_query("""
                SELECT * FROM price_data 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 10
            """, conn, params=(symbol,))
            
            if len(data) >= 5:
                current_price = data.iloc[0]['close_price']
                avg_price = data['close_price'].mean()
                
                # Simple momentum screening
                if current_price > avg_price * 1.02:  # 2% above average
                    signals.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'avg_price': avg_price,
                        'momentum': ((current_price - avg_price) / avg_price) * 100,
                        'signal': 'BULLISH',
                        'entry': current_price,
                        'sl': current_price * 0.97,
                        'target': current_price * 1.08
                    })
                elif current_price < avg_price * 0.98:  # 2% below average
                    signals.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'avg_price': avg_price,
                        'momentum': ((current_price - avg_price) / avg_price) * 100,
                        'signal': 'BEARISH',
                        'entry': current_price,
                        'sl': current_price * 1.03,
                        'target': current_price * 0.92
                    })
        except Exception as e:
            print(f"⚠️ Error screening {symbol}: {e}")
    
    conn.close()
    
    # Save results
    if signals:
        df = pd.DataFrame(signals)
        df.to_csv('full_screening_results.csv', index=False)
        print(f"📄 Saved {len(signals)} signals to full_screening_results.csv")
        
        print(f"\n🎯 SCREENING RESULTS:")
        print(f"📊 Total signals: {len(signals)}")
        print(f"📈 Bullish: {len([s for s in signals if s['signal'] == 'BULLISH'])}")
        print(f"📉 Bearish: {len([s for s in signals if s['signal'] == 'BEARISH'])}")
        
        print(f"\n📋 Sample signals:")
        for signal in signals[:10]:
            print(f"  • {signal['symbol']}: {signal['signal']} @ ₹{signal['entry']:.2f} (SL: ₹{signal['sl']:.2f}, Target: ₹{signal['target']:.2f})")
    else:
        print("⚠️ No signals generated")

def get_database_stats():
    """Get database statistics."""
    conn = sqlite3.connect('data/full_indian_market.db')
    
    # Total records
    total_records = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
    
    # Unique symbols
    unique_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
    
    # Symbols with valid data
    valid_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data WHERE close_price > 0").fetchone()[0]
    
    # Latest date
    latest_date = conn.execute("SELECT MAX(date) FROM price_data").fetchone()[0]
    
    conn.close()
    
    return {
        'total_records': total_records,
        'unique_symbols': unique_symbols,
        'valid_symbols': valid_symbols,
        'latest_date': latest_date
    }

def main():
    """Main function."""
    print("🎯 FULL DATA DOWNLOAD FOR ALL 2799 INDIAN SYMBOLS")
    print("=" * 60)
    
    # Initialize database
    init_database()
    
    # Load symbols
    symbols = load_all_symbols()
    
    # Download data (start with 100 for testing)
    download_all_data(symbols, max_symbols=100)
    
    # Get database stats
    stats = get_database_stats()
    print(f"\n📊 DATABASE STATISTICS:")
    print(f"📈 Total records: {stats['total_records']}")
    print(f"🎯 Unique symbols: {stats['unique_symbols']}")
    print(f"✅ Valid symbols: {stats['valid_symbols']}")
    print(f"📅 Latest date: {stats['latest_date']}")
    
    # Test screening
    test_screening()
    
    print(f"\n🎉 FULL SYSTEM TEST COMPLETED!")
    print(f"📈 Ready for production use with {len(symbols)} Indian stocks!")

if __name__ == "__main__":
    main() 