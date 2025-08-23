#!/usr/bin/env python3
"""
Test Screening with Database Only
Tests the screening system using only database data.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_database_screening():
    """Test screening using only database data."""
    print("🧪 TESTING SCREENING WITH DATABASE ONLY")
    print("=" * 50)
    
    # Connect to database
    conn = sqlite3.connect("data/comprehensive_equity.db")
    
    # Get database stats
    total_symbols = conn.execute('SELECT COUNT(*) FROM securities').fetchone()[0]
    total_records = conn.execute('SELECT COUNT(*) FROM price_data').fetchone()[0]
    
    print(f"📊 Database Statistics:")
    print(f"   📈 Total Symbols: {total_symbols}")
    print(f"   📊 Total Records: {total_records:,}")
    
    # Get sample symbols with good data
    print(f"\n🎯 Getting Sample Symbols...")
    
    # Get symbols with sufficient historical data
    symbols_data = conn.execute('''
        SELECT symbol, COUNT(*) as record_count 
        FROM price_data 
        GROUP BY symbol 
        HAVING COUNT(*) >= 50
        ORDER BY RANDOM() 
        LIMIT 20
    ''').fetchall()
    
    symbols = [row[0] for row in symbols_data]
    print(f"   🎯 Testing {len(symbols)} symbols with sufficient data")
    print(f"   📋 Sample: {symbols[:5]}")
    
    # Test screening for each symbol
    print(f"\n🔍 Running Database-Only Screening...")
    
    bullish_signals = []
    bearish_signals = []
    
    for i, symbol in enumerate(symbols):
        print(f"   📊 Processing {i+1}/{len(symbols)}: {symbol}")
        
        try:
            # Get historical data
            historical_data = pd.read_sql_query('''
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM price_data 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 100
            ''', conn, params=[symbol])
            
            if historical_data.empty or len(historical_data) < 20:
                continue
            
            # Process data
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            historical_data = historical_data.sort_values('date')
            historical_data.set_index('date', inplace=True)
            
            # Get current price (latest close)
            current_price = historical_data['close_price'].iloc[-1]
            current_volume = historical_data['volume'].iloc[-1]
            
            # Apply basic filters
            if current_volume < 100000 or current_price < 10.0:
                continue
            
            # Calculate indicators
            indicators = calculate_indicators(historical_data)
            
            if not indicators:
                continue
            
            # Generate signal
            signal = generate_signal(indicators, current_price)
            
            if signal['signal'] != 'NEUTRAL':
                # Calculate levels
                levels = calculate_levels(current_price, indicators)
                
                result = {
                    'symbol': symbol,
                    'signal': signal['signal'],
                    'confidence': signal['confidence'],
                    'reasons': signal['reasons'],
                    'entry_price': levels['entry'],
                    'stop_loss': levels['stop_loss'],
                    'targets': levels['targets'],
                    'risk_reward_ratio': levels['risk_reward'],
                    'current_price': current_price,
                    'volume': current_volume,
                    'screening_date': datetime.now().isoformat()
                }
                
                if signal['signal'] == 'BULLISH':
                    bullish_signals.append(result)
                else:
                    bearish_signals.append(result)
                
                print(f"      ✅ {signal['signal']} signal generated (Confidence: {signal['confidence']}%)")
            else:
                print(f"      ⚪ Neutral signal")
                
        except Exception as e:
            print(f"      ❌ Error processing {symbol}: {e}")
            continue
    
    conn.close()
    
    # Display results
    print(f"\n📊 SCREENING RESULTS:")
    print("=" * 40)
    
    print(f"📈 Summary:")
    print(f"   🎯 Total Screened: {len(symbols)}")
    print(f"   ✅ Bullish Signals: {len(bullish_signals)}")
    print(f"   ❌ Bearish Signals: {len(bearish_signals)}")
    print(f"   📅 Date: {datetime.now().isoformat()}")
    
    if bullish_signals:
        print(f"\n📈 BULLISH Signals ({len(bullish_signals)}):")
        for i, signal in enumerate(bullish_signals[:10]):
            print(f"   {i+1}. ✅ {signal['symbol']}: ₹{signal['current_price']:.2f} | Entry: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | Target: ₹{signal['targets']['T1']:.2f} | Confidence: {signal['confidence']}%")
        if len(bullish_signals) > 10:
            print(f"   ... and {len(bullish_signals) - 10} more bullish signals")
    
    if bearish_signals:
        print(f"\n📉 BEARISH Signals ({len(bearish_signals)}):")
        for i, signal in enumerate(bearish_signals[:10]):
            print(f"   {i+1}. ❌ {signal['symbol']}: ₹{signal['current_price']:.2f} | Entry: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | Target: ₹{signal['targets']['T1']:.2f} | Confidence: {signal['confidence']}%")
        if len(bearish_signals) > 10:
            print(f"   ... and {len(bearish_signals) - 10} more bearish signals")
    
    # Performance analysis
    if bullish_signals or bearish_signals:
        total_signals = len(bullish_signals) + len(bearish_signals)
        signal_rate = (total_signals / len(symbols)) * 100
        print(f"\n📊 Performance Analysis:")
        print(f"   🎯 Signal Rate: {signal_rate:.1f}%")
        print(f"   📈 Bullish Rate: {(len(bullish_signals)/len(symbols)*100):.1f}%")
        print(f"   📉 Bearish Rate: {(len(bearish_signals)/len(symbols)*100):.1f}%")
    
    return len(bullish_signals) + len(bearish_signals) > 0

def calculate_indicators(df):
    """Calculate technical indicators."""
    if df.empty or len(df) < 20:
        return {}
    
    indicators = {}
    
    # Moving averages
    indicators['sma_20'] = df['close_price'].rolling(20).mean().iloc[-1]
    if len(df) >= 50:
        indicators['sma_50'] = df['close_price'].rolling(50).mean().iloc[-1]
    
    # RSI
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    if loss > 0:
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    if len(df) >= 20:
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        if avg_volume > 0:
            indicators['volume_ratio'] = current_volume / avg_volume
    
    return indicators

def generate_signal(indicators, current_price):
    """Generate trading signal."""
    bullish_score = 0
    reasons = []
    
    # Bullish signals
    if 'sma_20' in indicators and current_price > indicators['sma_20']:
        bullish_score += 1
        reasons.append("Price above 20-day SMA")
    
    if 'sma_50' in indicators and indicators['sma_50'] and current_price > indicators['sma_50']:
        bullish_score += 1
        reasons.append("Price above 50-day SMA")
    
    if 'rsi' in indicators and 30 < indicators['rsi'] < 70:
        bullish_score += 1
        reasons.append("RSI in neutral zone")
    
    if 'volume_ratio' in indicators and indicators['volume_ratio'] > 1.5:
        bullish_score += 1
        reasons.append("High volume confirmation")
    
    # Bearish signals
    bearish_score = 0
    
    if 'sma_20' in indicators and current_price < indicators['sma_20']:
        bearish_score += 1
    
    if 'sma_50' in indicators and indicators['sma_50'] and current_price < indicators['sma_50']:
        bearish_score += 1
    
    if 'rsi' in indicators and indicators['rsi'] > 70:
        bearish_score += 1
    
    # Determine signal
    if bullish_score >= 3 and bullish_score > bearish_score:
        signal = 'BULLISH'
        confidence = min(90, 50 + (bullish_score * 15))
    elif bearish_score >= 3 and bearish_score > bullish_score:
        signal = 'BEARISH'
        confidence = min(90, 50 + (bearish_score * 15))
        reasons = [f"Bearish: {r}" for r in reasons]
    else:
        signal = 'NEUTRAL'
        confidence = 0
    
    return {
        'signal': signal,
        'confidence': confidence,
        'reasons': reasons
    }

def calculate_levels(current_price, indicators):
    """Calculate entry, stop loss, and target levels."""
    # Simple ATR calculation
    atr = current_price * 0.02  # 2% of current price as ATR approximation
    
    if 'sma_20' in indicators:
        # Use SMA for better levels
        sma_20 = indicators['sma_20']
        entry = current_price
        stop_loss = sma_20 * 0.95  # 5% below SMA
        target1 = entry + (entry - stop_loss) * 2  # 2:1 risk-reward
        target2 = entry + (entry - stop_loss) * 3  # 3:1 risk-reward
        target3 = entry + (entry - stop_loss) * 5  # 5:1 risk-reward
    else:
        # Fallback to simple levels
        entry = current_price
        stop_loss = entry * 0.95
        target1 = entry * 1.10
        target2 = entry * 1.20
        target3 = entry * 1.30
    
    risk = abs(entry - stop_loss)
    reward = abs(target1 - entry)
    risk_reward = reward / risk if risk > 0 else 0
    
    return {
        'entry': round(entry, 2),
        'stop_loss': round(stop_loss, 2),
        'targets': {
            'T1': round(target1, 2),
            'T2': round(target2, 2),
            'T3': round(target3, 2)
        },
        'risk_reward': round(risk_reward, 2)
    }

if __name__ == "__main__":
    print("🚀 STARTING DATABASE-ONLY SCREENING TEST")
    print("=" * 60)
    
    success = test_database_screening()
    
    if success:
        print(f"\n🎉 SCREENING TEST SUCCESSFUL!")
    else:
        print(f"\n⚠️ No signals generated - may need to adjust criteria")
    
    print("=" * 50)
