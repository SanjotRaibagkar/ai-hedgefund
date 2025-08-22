#!/usr/bin/env python3
"""
Enhanced EOD Screener
Fast EOD screening using database data with CSV output.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from loguru import logger
import os

# Import data manager
from src.data.indian_data_manager import indian_data_manager

class EnhancedEODScreener:
    """Enhanced EOD screener with database integration."""
    
    def __init__(self):
        self.db_path = indian_data_manager.db_path
        self.results_dir = "results/eod_screening"
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def screen_universe(self, 
                            symbols: List[str] = None,
                            start_date: str = None,
                            end_date: str = None,
                            min_volume: int = 100000,
                            min_price: float = 10.0,
                            max_workers: int = 20) -> Dict:
        """
        Screen entire universe of stocks.
        
        Args:
            symbols: List of symbols to screen. If None, screens all.
            start_date: Start date for analysis
            end_date: End date for analysis
            min_volume: Minimum average volume
            min_price: Minimum price filter
            max_workers: Maximum concurrent workers
        """
        logger.info("🎯 Starting Enhanced EOD Screening...")
        
        # Set default dates (last 6 months)
        if not start_date:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get symbols to screen
        if not symbols:
            symbols = await self._get_all_symbols()
        
        logger.info(f"📊 Screening {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Screen symbols concurrently
        start_time = datetime.now()
        results = await self._screen_symbols_concurrent(
            symbols, start_date, end_date, min_volume, min_price, max_workers
        )
        
        # Process results
        bullish_signals = []
        bearish_signals = []
        
        for result in results:
            if result['signal'] == 'BULLISH':
                bullish_signals.append(result)
            elif result['signal'] == 'BEARISH':
                bearish_signals.append(result)
        
        # Generate summary
        summary = {
            'total_screened': len(symbols),
            'bullish_signals': len(bullish_signals),
            'bearish_signals': len(bearish_signals),
            'screening_date': datetime.now().isoformat(),
            'date_range': {'start': start_date, 'end': end_date},
            'filters': {
                'min_volume': min_volume,
                'min_price': min_price
            }
        }
        
        # Save results to CSV
        await self._save_results_to_csv(bullish_signals, bearish_signals, summary)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Screening completed in {elapsed:.2f}s")
        logger.info(f"   📈 Bullish: {len(bullish_signals)}")
        logger.info(f"   📉 Bearish: {len(bearish_signals)}")
        
        return {
            'summary': summary,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
    
    async def _get_all_symbols(self) -> List[str]:
        """Get all symbols from database."""
        with sqlite3.connect(self.db_path) as conn:
            symbols = pd.read_sql_query(
                "SELECT symbol FROM securities WHERE is_active = 1", 
                conn
            )['symbol'].tolist()
        return symbols
    
    async def _screen_symbols_concurrent(self, 
                                       symbols: List[str],
                                       start_date: str,
                                       end_date: str,
                                       min_volume: int,
                                       min_price: float,
                                       max_workers: int) -> List[Dict]:
        """Screen symbols concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._screen_single_symbol, 
                    symbol, start_date, end_date, min_volume, min_price
                ): symbol
                for symbol in symbols
            }
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error screening {symbol}: {e}")
                
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"📊 Progress: {completed}/{len(symbols)}")
        
        return results
    
    def _screen_single_symbol(self, 
                            symbol: str,
                            start_date: str,
                            end_date: str,
                            min_volume: int,
                            min_price: float) -> Optional[Dict]:
        """Screen a single symbol."""
        try:
            # Get price data from database
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
            
            if df.empty or len(df) < 30:  # Need at least 30 days of data
                return None
            
            # Convert to proper format
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Apply filters
            avg_volume = df['volume'].mean()
            avg_price = df['close_price'].mean()
            
            if avg_volume < min_volume or avg_price < min_price:
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Generate signal
            signal = self._generate_signal(indicators, df)
            
            if signal['signal'] == 'NEUTRAL':
                return None
            
            # Calculate entry, SL, targets
            levels = self._calculate_levels(df, signal['signal'])
            
            return {
                'symbol': symbol,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'reasons': signal['reasons'],
                'entry_price': levels['entry'],
                'stop_loss': levels['stop_loss'],
                'targets': levels['targets'],
                'risk_reward_ratio': levels['risk_reward'],
                'current_price': df['close_price'].iloc[-1],
                'avg_volume': avg_volume,
                'avg_price': avg_price,
                'screening_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error screening {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = df['close_price'].rolling(20).mean()
        indicators['sma_50'] = df['close_price'].rolling(50).mean()
        indicators['ema_12'] = df['close_price'].ewm(span=12).mean()
        indicators['ema_26'] = df['close_price'].ewm(span=26).mean()
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        indicators['bb_middle'] = df['close_price'].rolling(20).mean()
        bb_std = df['close_price'].rolling(20).std()
        indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
        indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        indicators['volume_sma'] = df['volume'].rolling(20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        # Price action
        indicators['high_20'] = df['high_price'].rolling(20).max()
        indicators['low_20'] = df['low_price'].rolling(20).min()
        
        return indicators
    
    def _generate_signal(self, indicators: Dict, df: pd.DataFrame) -> Dict:
        """Generate trading signal based on indicators."""
        current_price = df['close_price'].iloc[-1]
        reasons = []
        confidence = 0
        
        # Check for bullish signals
        bullish_score = 0
        
        # Price above moving averages
        if current_price > indicators['sma_20'].iloc[-1]:
            bullish_score += 1
            reasons.append("Price above 20-day SMA")
        
        if current_price > indicators['sma_50'].iloc[-1]:
            bullish_score += 1
            reasons.append("Price above 50-day SMA")
        
        # MACD bullish
        if (indicators['macd'].iloc[-1] > indicators['macd_signal'].iloc[-1] and 
            indicators['macd_histogram'].iloc[-1] > indicators['macd_histogram'].iloc[-2]):
            bullish_score += 2
            reasons.append("MACD bullish crossover")
        
        # RSI oversold bounce
        if 30 < indicators['rsi'].iloc[-1] < 50:
            bullish_score += 1
            reasons.append("RSI oversold bounce")
        
        # Volume confirmation
        if indicators['volume_ratio'].iloc[-1] > 1.5:
            bullish_score += 1
            reasons.append("High volume confirmation")
        
        # Breakout from Bollinger Bands
        if current_price > indicators['bb_upper'].iloc[-1]:
            bullish_score += 2
            reasons.append("Breakout above Bollinger Bands")
        
        # Check for bearish signals
        bearish_score = 0
        
        # Price below moving averages
        if current_price < indicators['sma_20'].iloc[-1]:
            bearish_score += 1
        
        if current_price < indicators['sma_50'].iloc[-1]:
            bearish_score += 1
        
        # MACD bearish
        if (indicators['macd'].iloc[-1] < indicators['macd_signal'].iloc[-1] and 
            indicators['macd_histogram'].iloc[-1] < indicators['macd_histogram'].iloc[-2]):
            bearish_score += 2
        
        # RSI overbought
        if indicators['rsi'].iloc[-1] > 70:
            bearish_score += 1
        
        # Breakdown from Bollinger Bands
        if current_price < indicators['bb_lower'].iloc[-1]:
            bearish_score += 2
        
        # Determine signal
        if bullish_score >= 4 and bullish_score > bearish_score:
            signal = 'BULLISH'
            confidence = min(90, 50 + (bullish_score * 10))
        elif bearish_score >= 4 and bearish_score > bullish_score:
            signal = 'BEARISH'
            confidence = min(90, 50 + (bearish_score * 10))
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
    
    def _calculate_levels(self, df: pd.DataFrame, signal: str) -> Dict:
        """Calculate entry, stop loss, and target levels."""
        current_price = df['close_price'].iloc[-1]
        atr = self._calculate_atr(df, 14)
        
        if signal == 'BULLISH':
            entry = current_price
            stop_loss = entry - (atr * 2)
            target1 = entry + (atr * 3)
            target2 = entry + (atr * 5)
            target3 = entry + (atr * 8)
        else:  # BEARISH
            entry = current_price
            stop_loss = entry + (atr * 2)
            target1 = entry - (atr * 3)
            target2 = entry - (atr * 5)
            target3 = entry - (atr * 8)
        
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
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range."""
        high = df['high_price']
        low = df['low_price']
        close = df['close_price']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    async def _save_results_to_csv(self, 
                                 bullish_signals: List[Dict],
                                 bearish_signals: List[Dict],
                                 summary: Dict):
        """Save screening results to CSV files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save bullish signals
        if bullish_signals:
            bullish_df = pd.DataFrame(bullish_signals)
            bullish_file = f"{self.results_dir}/bullish_signals_{timestamp}.csv"
            bullish_df.to_csv(bullish_file, index=False)
            logger.info(f"📈 Bullish signals saved to {bullish_file}")
        
        # Save bearish signals
        if bearish_signals:
            bearish_df = pd.DataFrame(bearish_signals)
            bearish_file = f"{self.results_dir}/bearish_signals_{timestamp}.csv"
            bearish_df.to_csv(bearish_file, index=False)
            logger.info(f"📉 Bearish signals saved to {bearish_file}")
        
        # Save summary
        summary_file = f"{self.results_dir}/screening_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        logger.info(f"📊 Summary saved to {summary_file}")

# Global instance
enhanced_eod_screener = EnhancedEODScreener() 