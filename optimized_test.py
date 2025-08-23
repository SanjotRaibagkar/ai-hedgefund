#!/usr/bin/env python3
"""
Optimized Data Download and Screening Test
"""

from src.nsedata.NseUtility import NseUtils
import sqlite3
import os
from datetime import datetime
import time
import pandas as pd

def main():
    print("🚀 OPTIMIZED DATA DOWNLOAD AND SCREENING TEST")
    print("=" * 60)
    
    # Setup database
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/optimized_market.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            symbol TEXT,
            date TEXT,
            close_price REAL,
            UNIQUE(symbol, date)
        )
    ''')
    conn.commit()
    
    # Working symbols (known to work)
    working_symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 
        'BHARTIARTL', 'KOTAKBANK', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'SUNPHARMA'
    ]
    
    print(f"📊 Testing with {len(working_symbols)} working symbols")
    
    # Download data
    nse = NseUtils()
    successful = 0
    start_time = time.time()
    
    for symbol in working_symbols:
        try:
            price_info = nse.price_info(symbol)
            if price_info and price_info.get('LastTradedPrice', 0) > 0:
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
    total_time = time.time() - start_time
    print(f"\n📊 Downloaded {successful}/{len(working_symbols)} symbols in {total_time:.1f}s")
    
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
                
                # Calculate momentum
                momentum = ((current_price - avg_price) / avg_price) * 100
                
                # Screening logic
                if momentum > 1:  # 1% above average
                    signals.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'avg_price': avg_price,
                        'momentum': momentum,
                        'signal': 'BULLISH',
                        'entry': current_price,
                        'sl': current_price * 0.98,
                        'target': current_price * 1.05,
                        'reason': f'Momentum: {momentum:.1f}% above average'
                    })
                elif momentum < -1:  # 1% below average
                    signals.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'avg_price': avg_price,
                        'momentum': momentum,
                        'signal': 'BEARISH',
                        'entry': current_price,
                        'sl': current_price * 1.02,
                        'target': current_price * 0.95,
                        'reason': f'Momentum: {momentum:.1f}% below average'
                    })
        except Exception as e:
            print(f"⚠️ Error screening {symbol}: {e}")
    
    # Save results
    if signals:
        df = pd.DataFrame(signals)
        df.to_csv('optimized_screening_results.csv', index=False)
        print(f"📄 Saved {len(signals)} signals to optimized_screening_results.csv")
        
        print(f"\n🎯 SCREENING RESULTS:")
        print(f"📊 Total signals: {len(signals)}")
        print(f"📈 Bullish: {len([s for s in signals if s['signal'] == 'BULLISH'])}")
        print(f"📉 Bearish: {len([s for s in signals if s['signal'] == 'BEARISH'])}")
        
        print(f"\n📋 Sample signals:")
        for signal in signals[:5]:
            print(f"  • {signal['symbol']}: {signal['signal']} @ ₹{signal['entry']:.2f}")
            print(f"    SL: ₹{signal['sl']:.2f}, Target: ₹{signal['target']:.2f}")
            print(f"    Reason: {signal['reason']}")
    else:
        print("⚠️ No signals generated")
    
    conn.close()
    print("\n🎉 OPTIMIZED TEST COMPLETED!")

if __name__ == "__main__":
    main() 