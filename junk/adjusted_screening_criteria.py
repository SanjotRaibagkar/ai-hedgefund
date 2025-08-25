#!/usr/bin/env python3
"""
Adjusted Screening Criteria - Generate More Realistic Signals
Adjust the signal generation criteria to be more practical
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import pandas as pd
from datetime import datetime
from loguru import logger

from src.screening.unified_eod_screener import unified_eod_screener
from src.data.database.duckdb_manager import DatabaseManager

class AdjustedEODScreener:
    """EOD Screener with adjusted criteria for more realistic signals."""
    
    def __init__(self):
        self.db_path = "data/comprehensive_equity.duckdb"
    
    async def _get_all_symbols(self) -> list:
        """Get all available symbols from database."""
        db_manager = DatabaseManager()
        return db_manager.get_available_symbols()
    
    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data for symbol."""
        db_manager = DatabaseManager()
        return db_manager.get_price_data(symbol, start_date, end_date)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators with adjusted logic."""
        if len(df) < 50:
            return {}
        
        indicators = {}
        
        # Basic indicators
        indicators['sma_20'] = df['close_price'].rolling(window=20).mean()
        indicators['sma_50'] = df['close_price'].rolling(window=50).mean()
        
        # RSI with adjusted calculation
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume ratio (current volume vs 20-day average)
        indicators['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # MACD
        ema_12 = df['close_price'].ewm(span=12).mean()
        ema_26 = df['close_price'].ewm(span=26).mean()
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Bollinger Bands
        sma_20 = indicators['sma_20']
        std_20 = df['close_price'].rolling(window=20).std()
        indicators['bb_upper'] = sma_20 + (std_20 * 2)
        indicators['bb_lower'] = sma_20 - (std_20 * 2)
        
        return indicators
    
    def _generate_adjusted_signal(self, indicators: dict, df: pd.DataFrame) -> dict:
        """Generate trading signal with adjusted criteria."""
        if df.empty or len(df) < 50:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'reasons': [], 'bullish_score': 0, 'bearish_score': 0}
        
        current_price = df['close_price'].iloc[-1]
        reasons = []
        
        # ADJUSTED CRITERIA - More realistic thresholds
        bullish_score = 0
        bearish_score = 0
        
        # 1. Price vs Moving Averages (1 point each)
        if 'sma_20' in indicators and not indicators['sma_20'].empty:
            if current_price > indicators['sma_20'].iloc[-1]:
                bullish_score += 1
                reasons.append("Price above 20-day SMA")
            else:
                bearish_score += 1
        
        if 'sma_50' in indicators and not indicators['sma_50'].empty:
            if current_price > indicators['sma_50'].iloc[-1]:
                bullish_score += 1
                reasons.append("Price above 50-day SMA")
            else:
                bearish_score += 1
        
        # 2. RSI Analysis (1 point each)
        if 'rsi' in indicators and not indicators['rsi'].empty:
            rsi_value = indicators['rsi'].iloc[-1]
            if 30 < rsi_value < 70:  # Neutral zone
                bullish_score += 1
                reasons.append("RSI in neutral zone")
            elif rsi_value < 30:  # Oversold
                bullish_score += 1
                reasons.append("RSI oversold")
            elif rsi_value > 70:  # Overbought
                bearish_score += 1
        
        # 3. Volume Analysis (1 point)
        if 'volume_ratio' in indicators and not indicators['volume_ratio'].empty:
            vol_ratio = indicators['volume_ratio'].iloc[-1]
            if vol_ratio > 1.2:  # Lowered threshold
                bullish_score += 1
                reasons.append("Above average volume")
        
        # 4. MACD Analysis (1 point)
        if ('macd' in indicators and 'macd_signal' in indicators and 
            not indicators['macd'].empty and not indicators['macd_signal'].empty):
            if indicators['macd'].iloc[-1] > indicators['macd_signal'].iloc[-1]:
                bullish_score += 1
                reasons.append("MACD bullish")
            else:
                bearish_score += 1
        
        # 5. Price momentum (1 point)
        if len(df) >= 5:
            price_5d_ago = df['close_price'].iloc[-5]
            if current_price > price_5d_ago:
                bullish_score += 1
                reasons.append("5-day price momentum")
            else:
                bearish_score += 1
        
        # ADJUSTED SIGNAL GENERATION - Lower thresholds
        if bullish_score >= 3 and bullish_score > bearish_score:
            signal = 'BULLISH'
            confidence = min(85, 40 + (bullish_score * 10))
        elif bearish_score >= 3 and bearish_score > bullish_score:
            signal = 'BEARISH'
            confidence = min(85, 40 + (bearish_score * 10))
            reasons = [f"Bearish: {r}" for r in reasons]
        else:
            signal = 'NEUTRAL'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasons': reasons,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score
        }
    
    def _calculate_levels(self, df: pd.DataFrame, signal: str) -> dict:
        """Calculate entry, stop loss, and target levels."""
        current_price = df['close_price'].iloc[-1]
        
        # Simple ATR calculation
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['close_price'].shift())
        low_close = abs(df['low_price'] - df['close_price'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        
        if signal == 'BULLISH':
            entry = current_price
            stop_loss = entry - (atr * 1.5)  # Reduced from 2x ATR
            target1 = entry + (atr * 2.5)    # Reduced from 3x ATR
            target2 = entry + (atr * 4.0)    # Reduced from 5x ATR
            target3 = entry + (atr * 6.0)    # Reduced from 8x ATR
        else:  # BEARISH
            entry = current_price
            stop_loss = entry + (atr * 1.5)
            target1 = entry - (atr * 2.5)
            target2 = entry - (atr * 4.0)
            target3 = entry - (atr * 6.0)
        
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
            'risk_reward_ratio': round(risk_reward, 2)
        }
    
    async def screen_universe(self, symbols: list, min_volume: int = 10000, 
                            min_price: float = 1.0) -> dict:
        """Screen universe with adjusted criteria."""
        logger.info("🎯 Starting Adjusted EOD Screening...")
        logger.info(f"📊 Screening {len(symbols)} symbols")
        
        start_date = (datetime.now() - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        start_time = datetime.now()
        
        # Screen symbols
        results = await self._screen_symbols_concurrent(symbols, start_date, end_date, min_volume, min_price)
        
        # Process results
        bullish_signals = []
        bearish_signals = []
        
        for result in results:
            if result and result['signal'] != 'NEUTRAL':
                signal_data = {
                    'symbol': result['symbol'],
                    'signal': result['signal'],
                    'confidence': result['confidence'],
                    'entry_price': result['levels']['entry'],
                    'stop_loss': result['levels']['stop_loss'],
                    'targets': result['levels']['targets'],
                    'risk_reward_ratio': result['levels']['risk_reward_ratio'],
                    'reasons': result['reasons'],
                    'current_price': result['current_price'],
                    'volume': result['volume']
                }
                
                if result['signal'] == 'BULLISH':
                    bullish_signals.append(signal_data)
                else:
                    bearish_signals.append(signal_data)
        
        # Sort by confidence
        bullish_signals.sort(key=lambda x: x['confidence'], reverse=True)
        bearish_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Screening completed in {duration:.2f}s")
        logger.info(f"   📈 Bullish: {len(bullish_signals)}")
        logger.info(f"   📉 Bearish: {len(bearish_signals)}")
        
        return {
            'summary': {
                'total_screened': len(symbols),
                'bullish_signals': len(bullish_signals),
                'bearish_signals': len(bearish_signals),
                'duration_seconds': duration
            },
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
    
    async def _screen_symbols_concurrent(self, symbols: list, start_date: str, 
                                       end_date: str, min_volume: int, min_price: float) -> list:
        """Screen symbols concurrently."""
        import concurrent.futures
        
        results = []
        max_workers = 10
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for symbol in symbols:
                future = executor.submit(self._screen_single_symbol, symbol, start_date, end_date, min_volume, min_price)
                futures.append(future)
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    completed += 1
                    if completed % 50 == 0:
                        logger.info(f"📊 Progress: {completed}/{len(symbols)}")
                except Exception as e:
                    logger.error(f"Error processing symbol: {e}")
                    completed += 1
        
        return results
    
    def _screen_single_symbol(self, symbol: str, start_date: str, end_date: str, 
                            min_volume: int, min_price: float) -> dict:
        """Screen single symbol."""
        try:
            df = self._get_historical_data(symbol, start_date, end_date)
            
            if df.empty or len(df) < 50:
                return None
            
            # Check basic criteria
            latest = df.iloc[-1]
            if latest['volume'] < min_volume or latest['close_price'] < min_price:
                return None
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            if not indicators:
                return None
            
            # Generate signal
            signal_result = self._generate_adjusted_signal(indicators, df)
            
            if signal_result['signal'] == 'NEUTRAL':
                return None
            
            # Calculate levels
            levels = self._calculate_levels(df, signal_result['signal'])
            
            return {
                'symbol': symbol,
                'signal': signal_result['signal'],
                'confidence': signal_result['confidence'],
                'reasons': signal_result['reasons'],
                'current_price': latest['close_price'],
                'volume': latest['volume'],
                'levels': levels
            }
            
        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}")
            return None

async def run_adjusted_screening():
    """Run screening with adjusted criteria."""
    logger.info("🚀 Running Adjusted EOD Screening")
    logger.info("=" * 50)
    
    try:
        # Initialize screener
        screener = AdjustedEODScreener()
        
        # Get all symbols
        all_symbols = await screener._get_all_symbols()
        logger.info(f"📊 Total symbols available: {len(all_symbols)}")
        
        # Run screening
        results = await screener.screen_universe(
            symbols=all_symbols,
            min_volume=10000,  # Lower volume threshold
            min_price=1.0      # Lower price threshold
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if results['bullish_signals']:
            bullish_df = pd.DataFrame(results['bullish_signals'])
            bullish_file = f"results/adjusted_bullish_signals_{timestamp}.csv"
            os.makedirs('results', exist_ok=True)
            bullish_df.to_csv(bullish_file, index=False)
            logger.info(f"📈 Bullish signals saved to: {bullish_file}")
        
        if results['bearish_signals']:
            bearish_df = pd.DataFrame(results['bearish_signals'])
            bearish_file = f"results/adjusted_bearish_signals_{timestamp}.csv"
            bearish_df.to_csv(bearish_file, index=False)
            logger.info(f"📉 Bearish signals saved to: {bearish_file}")
        
        # Show summary
        summary = results['summary']
        logger.info("\n📊 ADJUSTED SCREENING SUMMARY")
        logger.info("=" * 40)
        logger.info(f"Total Symbols: {summary['total_screened']}")
        logger.info(f"Bullish Signals: {summary['bullish_signals']}")
        logger.info(f"Bearish Signals: {summary['bearish_signals']}")
        logger.info(f"Total Signals: {summary['bullish_signals'] + summary['bearish_signals']}")
        logger.info(f"Duration: {summary['duration_seconds']:.2f}s")
        
        # Show top signals
        if results['bullish_signals']:
            logger.info("\n📈 TOP 5 BULLISH SIGNALS:")
            for i, signal in enumerate(results['bullish_signals'][:5], 1):
                logger.info(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence")
                logger.info(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}")
                logger.info(f"   Risk-Reward: {signal['risk_reward_ratio']}")
                logger.info(f"   Reasons: {', '.join(signal['reasons'][:3])}")
                logger.info("")
        
        if results['bearish_signals']:
            logger.info("\n📉 TOP 5 BEARISH SIGNALS:")
            for i, signal in enumerate(results['bearish_signals'][:5], 1):
                logger.info(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence")
                logger.info(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}")
                logger.info(f"   Risk-Reward: {signal['risk_reward_ratio']}")
                logger.info(f"   Reasons: {', '.join(signal['reasons'][:3])}")
                logger.info("")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in adjusted screening: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(run_adjusted_screening())
