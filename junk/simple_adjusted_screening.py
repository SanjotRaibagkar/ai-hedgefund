#!/usr/bin/env python3
"""
Simple Adjusted Screening - Generate More Realistic Signals
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import pandas as pd
from datetime import datetime
from loguru import logger

from src.data.database.duckdb_manager import DatabaseManager

async def run_simple_adjusted_screening():
    """Run simple screening with adjusted criteria."""
    logger.info("🚀 Running Simple Adjusted Screening")
    logger.info("=" * 50)
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        all_symbols = db_manager.get_available_symbols()
        logger.info(f"📊 Total symbols: {len(all_symbols)}")
        
        # Test with first 100 symbols
        test_symbols = all_symbols[:100]
        logger.info(f"🔍 Testing with first 100 symbols")
        
        start_date = (datetime.now() - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        bullish_signals = []
        bearish_signals = []
        
        for i, symbol in enumerate(test_symbols):
            if i % 10 == 0:
                logger.info(f"📊 Progress: {i}/{len(test_symbols)}")
            
            try:
                # Get data
                df = db_manager.get_price_data(symbol, start_date, end_date)
                
                if df.empty or len(df) < 50:
                    continue
                
                latest = df.iloc[-1]
                
                # Simple criteria
                current_price = latest['close_price']
                volume = latest['volume']
                
                # Calculate simple indicators
                sma_20 = df['close_price'].rolling(20).mean().iloc[-1]
                sma_50 = df['close_price'].rolling(50).mean().iloc[-1]
                
                # Simple RSI
                delta = df['close_price'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
                rs = gain / loss if loss != 0 else 0
                rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
                
                # Simple signal generation
                bullish_score = 0
                bearish_score = 0
                reasons = []
                
                # Price vs moving averages
                if current_price > sma_20:
                    bullish_score += 1
                    reasons.append("Above SMA20")
                else:
                    bearish_score += 1
                
                if current_price > sma_50:
                    bullish_score += 1
                    reasons.append("Above SMA50")
                else:
                    bearish_score += 1
                
                # RSI analysis
                if 30 < rsi < 70:
                    bullish_score += 1
                    reasons.append("RSI neutral")
                elif rsi < 30:
                    bullish_score += 1
                    reasons.append("RSI oversold")
                elif rsi > 70:
                    bearish_score += 1
                
                # Volume check
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                if volume > avg_volume * 1.1:
                    bullish_score += 1
                    reasons.append("High volume")
                
                # Generate signal
                if bullish_score >= 3 and bullish_score > bearish_score:
                    signal = 'BULLISH'
                    confidence = min(80, 40 + (bullish_score * 10))
                elif bearish_score >= 3 and bearish_score > bullish_score:
                    signal = 'BEARISH'
                    confidence = min(80, 40 + (bearish_score * 10))
                else:
                    continue
                
                # Calculate levels
                atr = df['high_price'].rolling(14).max() - df['low_price'].rolling(14).min()
                atr = atr.mean()
                
                if signal == 'BULLISH':
                    entry = current_price
                    stop_loss = entry - (atr * 1.5)
                    target1 = entry + (atr * 2.5)
                else:
                    entry = current_price
                    stop_loss = entry + (atr * 1.5)
                    target1 = entry - (atr * 2.5)
                
                risk = abs(entry - stop_loss)
                reward = abs(target1 - entry)
                risk_reward = reward / risk if risk > 0 else 0
                
                signal_data = {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target1': round(target1, 2),
                    'risk_reward_ratio': round(risk_reward, 2),
                    'reasons': reasons,
                    'current_price': current_price,
                    'volume': volume
                }
                
                if signal == 'BULLISH':
                    bullish_signals.append(signal_data)
                else:
                    bearish_signals.append(signal_data)
                
            except Exception as e:
                continue
        
        # Sort by confidence
        bullish_signals.sort(key=lambda x: x['confidence'], reverse=True)
        bearish_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"\n📊 RESULTS SUMMARY")
        logger.info("=" * 30)
        logger.info(f"Symbols tested: {len(test_symbols)}")
        logger.info(f"Bullish signals: {len(bullish_signals)}")
        logger.info(f"Bearish signals: {len(bearish_signals)}")
        logger.info(f"Total signals: {len(bullish_signals) + len(bearish_signals)}")
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if bullish_signals:
            bullish_df = pd.DataFrame(bullish_signals)
            bullish_file = f"results/simple_bullish_signals_{timestamp}.csv"
            os.makedirs('results', exist_ok=True)
            bullish_df.to_csv(bullish_file, index=False)
            logger.info(f"📈 Bullish signals saved to: {bullish_file}")
        
        if bearish_signals:
            bearish_df = pd.DataFrame(bearish_signals)
            bearish_file = f"results/simple_bearish_signals_{timestamp}.csv"
            bearish_df.to_csv(bearish_file, index=False)
            logger.info(f"📉 Bearish signals saved to: {bearish_file}")
        
        # Show top signals
        if bullish_signals:
            logger.info("\n📈 TOP 5 BULLISH SIGNALS:")
            for i, signal in enumerate(bullish_signals[:5], 1):
                logger.info(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence")
                logger.info(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['target1']}")
                logger.info(f"   Risk-Reward: {signal['risk_reward_ratio']}")
                logger.info(f"   Reasons: {', '.join(signal['reasons'])}")
                logger.info("")
        
        if bearish_signals:
            logger.info("\n📉 TOP 5 BEARISH SIGNALS:")
            for i, signal in enumerate(bearish_signals[:5], 1):
                logger.info(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence")
                logger.info(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['target1']}")
                logger.info(f"   Risk-Reward: {signal['risk_reward_ratio']}")
                logger.info(f"   Reasons: {', '.join(signal['reasons'])}")
                logger.info("")
        
        return {
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(run_simple_adjusted_screening())
