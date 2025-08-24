#!/usr/bin/env python3
"""
DuckDB EOD Screener
Fast EOD screening using NSEUtility and DuckDB.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb
from loguru import logger
import os

class DuckDBEODScreener:
    """DuckDB EOD screener using NSEUtility and DuckDB."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        self.db_path = db_path
        self.results_dir = "results/eod_screening"
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def screen_universe(self, 
                            symbols: List[str] = None,
                            min_volume: int = 100000,
                            min_price: float = 10.0) -> Dict:
        """Screen universe of stocks."""
        logger.info("🎯 Starting DuckDB EOD Screening...")
        
        # Get symbols to screen
        if not symbols:
            symbols = await self._get_all_symbols()
        
        logger.info(f"📊 Screening {len(symbols)} symbols")
        
        # Screen symbols concurrently
        start_time = datetime.now()
        results = await self._screen_symbols_concurrent(symbols, min_volume, min_price)
        
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
        try:
            with duckdb.connect(self.db_path) as conn:
                # Check if securities table exists
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [table[0] for table in tables]
                
                if 'securities' in table_names:
                    # Check if securities table has data
                    securities_count = conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
                    if securities_count > 0:
                        symbols = pd.read_sql_query(
                            "SELECT symbol FROM securities WHERE is_active = true", 
                            conn
                        )['symbol'].tolist()
                    else:
                        # Securities table exists but is empty, use price_data
                        symbols = pd.read_sql_query(
                            "SELECT DISTINCT symbol FROM price_data", 
                            conn
                        )['symbol'].tolist()
                else:
                    # Fallback to price_data table
                    symbols = pd.read_sql_query(
                        "SELECT DISTINCT symbol FROM price_data", 
                        conn
                    )['symbol'].tolist()
                
                return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []
    
    async def _screen_symbols_concurrent(self, 
                                       symbols: List[str],
                                       min_volume: int,
                                       min_price: float) -> List[Dict]:
        """Screen symbols concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {
                executor.submit(
                    self._screen_single_symbol, 
                    symbol, min_volume, min_price
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
                if completed % 50 == 0:
                    logger.info(f"📊 Progress: {completed}/{len(symbols)}")
        
        return results
    
    def _screen_single_symbol(self, 
                            symbol: str,
                            min_volume: int,
                            min_price: float) -> Optional[Dict]:
        """Screen a single symbol using existing NSEUtility."""
        try:
            # Get current data from existing NSEUtility
            from src.nsedata.NseUtility import NseUtils
            nse = NseUtils()
            
            price_info = nse.price_info(symbol)
            if not price_info:
                return None
            
            current_price = float(price_info.get('LastTradedPrice', 0))
            volume = int(price_info.get('Volume', 0))
            
            # Apply filters
            if volume < min_volume or current_price < min_price:
                return None
            
            # Get historical data from database
            historical_data = self._get_historical_data(symbol)
            
            if historical_data.empty:
                return None
            
            # Calculate indicators
            indicators = self._calculate_indicators(historical_data)
            
            # Generate signal
            signal = self._generate_signal(indicators, current_price)
            
            if signal['signal'] == 'NEUTRAL':
                return None
            
            # Calculate levels
            levels = self._calculate_levels(current_price, indicators)
            
            return {
                'symbol': symbol,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'reasons': signal['reasons'],
                'entry_price': levels['entry'],
                'stop_loss': levels['stop_loss'],
                'targets': levels['targets'],
                'risk_reward_ratio': levels['risk_reward'],
                'current_price': current_price,
                'volume': volume,
                'screening_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error screening {symbol}: {e}")
            return None
    
    def _get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data from database."""
        try:
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 100
            """
            
            with duckdb.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[symbol])
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()  # Sort by date ascending
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        indicators = {}
        
        if df.empty or len(df) < 20:
            return indicators
        
        # Moving averages
        indicators['sma_20'] = df['close_price'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = df['close_price'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
        
        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        if loss > 0:
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        else:
            indicators['rsi'] = 50
        
        # Volume analysis
        indicators['avg_volume'] = df['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['avg_volume'] if indicators['avg_volume'] > 0 else 1
        
        # Price action
        indicators['high_20'] = df['high_price'].rolling(20).max().iloc[-1]
        indicators['low_20'] = df['low_price'].rolling(20).min().iloc[-1]
        
        return indicators
    
    def _generate_signal(self, indicators: Dict, current_price: float) -> Dict:
        """Generate trading signal."""
        reasons = []
        confidence = 0
        
        # Bullish signals
        bullish_score = 0
        
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
    
    def _calculate_levels(self, current_price: float, indicators: Dict) -> Dict:
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
duckdb_eod_screener = DuckDBEODScreener() 