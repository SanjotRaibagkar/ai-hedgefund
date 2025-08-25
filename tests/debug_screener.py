#!/usr/bin/env python3
"""
Debug Screener
Debug why the screener isn't generating signals.
"""

import asyncio
from src.screening.duckdb_eod_screener import duckdb_eod_screener
from src.nsedata.NseUtility import NseUtils
import pandas as pd

async def debug_screener():
    """Debug the screener step by step."""
    print("🔍 DEBUGGING SCREENER")
    print("=" * 40)
    
    screener = duckdb_eod_screener
    symbol = "RELIANCE"
    
    print(f"🎯 Testing symbol: {symbol}")
    
    # Step 1: Test NSEUtility
    print(f"\n1️⃣ Testing NSEUtility...")
    nse = NseUtils()
    price_info = nse.price_info(symbol)
    
    if price_info:
        print(f"   ✅ Price info available:")
        print(f"      Last Price: ₹{price_info.get('LastTradedPrice', 'N/A')}")
        print(f"      Volume: {price_info.get('Volume', 'N/A')}")
        print(f"      Open: ₹{price_info.get('Open', 'N/A')}")
        print(f"      High: ₹{price_info.get('High', 'N/A')}")
        print(f"      Low: ₹{price_info.get('Low', 'N/A')}")
    else:
        print(f"   ❌ No price info available")
        return
    
    # Step 2: Test historical data
    print(f"\n2️⃣ Testing historical data...")
    historical_data = screener._get_historical_data(symbol)
    
    if not historical_data.empty:
        print(f"   ✅ Historical data available:")
        print(f"      Records: {len(historical_data)}")
        print(f"      Date range: {historical_data.index.min()} to {historical_data.index.max()}")
        print(f"      Latest close: ₹{historical_data['close_price'].iloc[-1]:.2f}")
        print(f"      Sample data:")
        print(historical_data[['close_price', 'volume']].head(3))
    else:
        print(f"   ❌ No historical data available")
        return
    
    # Step 3: Test indicators
    print(f"\n3️⃣ Testing indicators...")
    indicators = screener._calculate_indicators(historical_data)
    
    if indicators:
        print(f"   ✅ Indicators calculated:")
        for key, value in indicators.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.2f}")
            else:
                print(f"      {key}: {value}")
    else:
        print(f"   ❌ No indicators calculated")
        return
    
    # Step 4: Test signal generation
    print(f"\n4️⃣ Testing signal generation...")
    current_price = float(price_info.get('LastTradedPrice', 0))
    signal = screener._generate_signal(indicators, current_price)
    
    print(f"   ✅ Signal generated:")
    print(f"      Signal: {signal['signal']}")
    print(f"      Confidence: {signal['confidence']}%")
    print(f"      Reasons: {signal['reasons']}")
    
    # Step 5: Test full screening
    print(f"\n5️⃣ Testing full screening...")
    result = screener._screen_single_symbol(symbol, min_volume=100000, min_price=10.0)
    
    if result:
        print(f"   ✅ Screening result:")
        print(f"      Symbol: {result['symbol']}")
        print(f"      Signal: {result['signal']}")
        print(f"      Confidence: {result['confidence']}%")
        print(f"      Entry: ₹{result['entry_price']}")
        print(f"      Stop Loss: ₹{result['stop_loss']}")
        print(f"      Target: ₹{result['targets']['T1']}")
    else:
        print(f"   ❌ No screening result")
        
        # Check why it failed
        volume = int(price_info.get('Volume', 0))
        print(f"   🔍 Debug info:")
        print(f"      Volume: {volume} (min required: 100000)")
        print(f"      Price: ₹{current_price} (min required: 10.0)")
        print(f"      Volume filter passed: {volume >= 100000}")
        print(f"      Price filter passed: {current_price >= 10.0}")

if __name__ == "__main__":
    asyncio.run(debug_screener())
