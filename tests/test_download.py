#!/usr/bin/env python3
"""
Simple test for data download and screening
"""

from src.nsedata.NseUtility import NseUtils
import sqlite3
import os
from datetime import datetime

def main():
    print("🚀 TESTING DATA DOWNLOAD AND SCREENING")
    print("=" * 50)
    
    # Setup database
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/test_market.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            symbol TEXT,
            date TEXT,
            close_price REAL,
            UNIQUE(symbol, date)
        )
    ''')
    conn.commit()
    
    # Test symbols
    test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK']
    
    print(f"📊 Testing with {len(test_symbols)} symbols")
    
    # Download data
    nse = NseUtils()
    successful = 0
    
    for symbol in test_symbols:
        try:
            price_info = nse.price_info(symbol)
            if price_info:
                price = price_info.get('LastTradedPrice', 0)
                conn.execute('INSERT OR REPLACE INTO price_data VALUES (?, ?, ?)', 
                           (symbol, datetime.now().strftime('%Y-%m-%d'), price))
                successful += 1
                print(f"✅ {symbol}: ₹{price}")
            else:
                print(f"❌ {symbol}: No data")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    conn.commit()
    print(f"\n📊 Downloaded {successful}/{len(test_symbols)} symbols")
    
    # Test screening
    print("\n🧪 Testing screening system...")
    
    # Get symbols with data
    symbols_with_data = conn.execute("SELECT DISTINCT symbol FROM price_data WHERE close_price > 0").fetchall()
    
    signals = []
    for (symbol,) in symbols_with_data:
        try:
            # Get recent prices
            prices = conn.execute("SELECT close_price FROM price_data WHERE symbol = ? ORDER BY date DESC LIMIT 5", (symbol,)).fetchall()
            
            if len(prices) >= 3:
                current_price = prices[0][0]
                avg_price = sum(row[0] for row in prices) / len(prices)
                
                if current_price > avg_price * 1.01:  # 1% above average
                    signals.append({
                        'symbol': symbol,
                        'price': current_price,
                        'signal': 'BULLISH',
                        'entry': current_price,
                        'sl': current_price * 0.98,
                        'target': current_price * 1.05
                    })
                elif current_price < avg_price * 0.99:  # 1% below average
                    signals.append({
                        'symbol': symbol,
                        'price': current_price,
                        'signal': 'BEARISH',
                        'entry': current_price,
                        'sl': current_price * 1.02,
                        'target': current_price * 0.95
                    })
        except Exception as e:
            print(f"⚠️ Error screening {symbol}: {e}")
    
    # Save results
    if signals:
        import pandas as pd
        df = pd.DataFrame(signals)
        df.to_csv('test_screening_results.csv', index=False)
        print(f"📄 Saved {len(signals)} signals to test_screening_results.csv")
        
        print(f"\n📋 Screening results:")
        for signal in signals:
            print(f"  • {signal['symbol']}: {signal['signal']} @ ₹{signal['entry']:.2f} (SL: ₹{signal['sl']:.2f}, Target: ₹{signal['target']:.2f})")
    else:
        print("⚠️ No signals generated")
    
    conn.close()
    print("\n🎉 TEST COMPLETED!")

if __name__ == "__main__":
    main() 