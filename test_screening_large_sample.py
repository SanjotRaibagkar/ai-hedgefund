#!/usr/bin/env python3
"""
Test Screening with Large Sample
Tests the screening system with a larger sample to generate more signals.
"""

import asyncio
import sqlite3
import time
from src.screening.simple_eod_screener import SimpleEODScreener

async def test_large_sample_screening():
    """Test screening with a larger sample."""
    print("🧪 TESTING SCREENING WITH LARGE SAMPLE")
    print("=" * 50)
    
    # Initialize screener
    screener = SimpleEODScreener()
    
    # Get larger sample of symbols
    print("🎯 Getting Large Sample of Symbols...")
    
    conn = sqlite3.connect("data/comprehensive_equity.db")
    
    # Get 50 random symbols
    symbols_data = conn.execute('''
        SELECT symbol FROM securities 
        ORDER BY RANDOM() 
        LIMIT 50
    ''').fetchall()
    
    symbols = [row[0] for row in symbols_data]
    print(f"   🎯 Testing {len(symbols)} symbols")
    print(f"   📋 Sample symbols: {symbols[:10]}...")  # Show first 10
    
    conn.close()
    
    # Test screening
    print(f"\n🔍 Running EOD Screening...")
    start_time = time.time()
    
    try:
        # Run screening
        results = await screener.screen_universe(symbols=symbols)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Screening completed in {duration:.2f} seconds")
        print(f"📊 Processing rate: {len(symbols)/duration:.1f} symbols/second")
        
        # Display results
        print(f"\n📊 SCREENING RESULTS:")
        print("=" * 40)
        
        summary = results['summary']
        bullish = results['bullish_signals']
        bearish = results['bearish_signals']
        
        print(f"📈 Summary:")
        print(f"   🎯 Total Screened: {summary['total_screened']}")
        print(f"   ✅ Bullish Signals: {summary['bullish_signals']}")
        print(f"   ❌ Bearish Signals: {summary['bearish_signals']}")
        print(f"   📅 Date: {summary['screening_date']}")
        
        if bullish:
            print(f"\n📈 BULLISH Signals ({len(bullish)}):")
            for i, signal in enumerate(bullish[:10]):  # Show first 10
                print(f"   {i+1}. ✅ {signal['symbol']}: ₹{signal.get('current_price', 'N/A')} | Entry: ₹{signal.get('entry_price', 'N/A')} | SL: ₹{signal.get('stop_loss', 'N/A')} | Target: ₹{signal.get('targets', {}).get('T1', 'N/A')} | Confidence: {signal.get('confidence', 'N/A')}%")
            if len(bullish) > 10:
                print(f"   ... and {len(bullish) - 10} more bullish signals")
        
        if bearish:
            print(f"\n📉 BEARISH Signals ({len(bearish)}):")
            for i, signal in enumerate(bearish[:10]):  # Show first 10
                print(f"   {i+1}. ❌ {signal['symbol']}: ₹{signal.get('current_price', 'N/A')} | Entry: ₹{signal.get('entry_price', 'N/A')} | SL: ₹{signal.get('stop_loss', 'N/A')} | Target: ₹{signal.get('targets', {}).get('T1', 'N/A')} | Confidence: {signal.get('confidence', 'N/A')}%")
            if len(bearish) > 10:
                print(f"   ... and {len(bearish) - 10} more bearish signals")
        
        # Performance analysis
        if bullish or bearish:
            total_signals = len(bullish) + len(bearish)
            signal_rate = (total_signals / len(symbols)) * 100
            print(f"\n📊 Performance Analysis:")
            print(f"   🎯 Signal Rate: {signal_rate:.1f}%")
            print(f"   📈 Bullish Rate: {(len(bullish)/len(symbols)*100):.1f}%")
            print(f"   📉 Bearish Rate: {(len(bearish)/len(symbols)*100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during screening: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_fno_stocks():
    """Test screening specifically with FNO stocks."""
    print(f"\n🔥 TESTING FNO STOCKS SCREENING")
    print("=" * 40)
    
    screener = SimpleEODScreener()
    
    # Get FNO stocks
    conn = sqlite3.connect("data/comprehensive_equity.db")
    
    fno_symbols_data = conn.execute('''
        SELECT symbol FROM securities 
        WHERE is_fno = 1 
        ORDER BY RANDOM() 
        LIMIT 20
    ''').fetchall()
    
    fno_symbols = [row[0] for row in fno_symbols_data]
    print(f"   🔥 Testing {len(fno_symbols)} FNO symbols")
    print(f"   📋 FNO symbols: {fno_symbols[:10]}...")
    
    conn.close()
    
    start_time = time.time()
    
    try:
        results = await screener.screen_universe(symbols=fno_symbols)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ FNO screening completed in {duration:.2f} seconds")
        
        summary = results['summary']
        bullish = results['bullish_signals']
        bearish = results['bearish_signals']
        
        print(f"📊 FNO Results:")
        print(f"   🎯 Total Screened: {summary['total_screened']}")
        print(f"   ✅ Bullish: {summary['bullish_signals']}")
        print(f"   ❌ Bearish: {summary['bearish_signals']}")
        
        if bullish:
            print(f"   📈 FNO Bullish Signals:")
            for signal in bullish[:5]:
                print(f"      ✅ {signal['symbol']}: ₹{signal.get('current_price', 'N/A')} | Confidence: {signal.get('confidence', 'N/A')}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during FNO screening: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING LARGE SAMPLE SCREENING TESTS")
    print("=" * 60)
    
    # Test large sample
    success1 = asyncio.run(test_large_sample_screening())
    
    if success1:
        # Test FNO stocks
        success2 = asyncio.run(test_fno_stocks())
    
    print(f"\n🎉 LARGE SAMPLE TESTING COMPLETED!")
    print("=" * 50)
