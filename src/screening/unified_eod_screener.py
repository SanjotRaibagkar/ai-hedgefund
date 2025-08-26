#!/usr/bin/env python3
"""
Unified EOD Screener
Combines the best features of SimpleEODScreener, EnhancedEODScreener, and DuckDBEODScreener
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb
from loguru import logger
import os

class UnifiedEODScreener:
    """Unified EOD screener with DuckDB integration and comprehensive analysis."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        self.db_path = db_path
        self.results_dir = "results/eod_screening"
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def screen_universe(self, 
                            symbols: List[str] = None,
                            start_date: str = None,
                            end_date: str = None,
                            min_volume: int = 100000,
                            min_price: float = 10.0,
                            max_workers: int = 20,
                            analysis_mode: str = "comprehensive") -> Dict:
        """
        Screen entire universe of stocks.
        
        Args:
            symbols: List of symbols to screen. If None, screens all.
            start_date: Start date for analysis
            end_date: End date for analysis
            min_volume: Minimum average volume
            min_price: Minimum price filter
            max_workers: Maximum concurrent workers
            analysis_mode: "basic", "enhanced", or "comprehensive"
        """
        logger.info("🎯 Starting Unified EOD Screening...")
        logger.info(f"📊 Analysis Mode: {analysis_mode}")
        
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
            symbols, start_date, end_date, min_volume, min_price, max_workers, analysis_mode
        )
        
        # Debug: Show raw results before filtering
        raw_bullish = len([r for r in results if r and r.get('signal') == 'BULLISH'])
        raw_bearish = len([r for r in results if r and r.get('signal') == 'BEARISH'])
        logger.info(f"📊 Raw Results (before filtering):")
        logger.info(f"   Raw bullish signals: {raw_bullish}")
        logger.info(f"   Raw bearish signals: {raw_bearish}")
        logger.info(f"   Total results: {len(results)}")
        
        # Process results with quality filters
        bullish_signals = []
        bearish_signals = []
        
        # Debug counters
        total_signals = 0
        confidence_filtered = 0
        risk_reward_filtered = 0
        volume_filtered = 0
        passed_filters = 0
        
        for result in results:
            if result is None:
                continue
                
            total_signals += 1
            
            # Quality filters (lowered thresholds)
            min_confidence = 50  # Lowered from 60%
            min_risk_reward = 1.5  # Lowered from 1.8
            min_volume_ratio = 1.0  # Lowered from 1.2
            
            # Check if signal meets quality criteria
            confidence_ok = result.get('confidence', 0) >= min_confidence
            risk_reward_ok = result.get('risk_reward_ratio', 0) >= min_risk_reward
            volume_ok = result.get('volume_ratio', 0) >= min_volume_ratio
            
            if not confidence_ok:
                confidence_filtered += 1
            if not risk_reward_ok:
                risk_reward_filtered += 1
            if not volume_ok:
                volume_filtered += 1
            
            if confidence_ok and risk_reward_ok and volume_ok:
                passed_filters += 1
                if result['signal'] == 'BULLISH':
                    bullish_signals.append(result)
                elif result['signal'] == 'BEARISH':
                    bearish_signals.append(result)
        
        # Log filter statistics
        logger.info(f"📊 Filter Statistics:")
        logger.info(f"   Total signals generated: {total_signals}")
        logger.info(f"   Confidence filtered: {confidence_filtered}")
        logger.info(f"   Risk-reward filtered: {risk_reward_filtered}")
        logger.info(f"   Volume filtered: {volume_filtered}")
        logger.info(f"   Passed all filters: {passed_filters}")
        
        # Generate summary
        summary = {
            'total_screened': len(symbols),
            'bullish_signals': len(bullish_signals),
            'bearish_signals': len(bearish_signals),
            'screening_date': datetime.now().isoformat(),
            'date_range': {'start': start_date, 'end': end_date},
            'analysis_mode': analysis_mode,
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
                symbols = conn.execute(
                    "SELECT DISTINCT symbol FROM price_data"
                ).fetchdf()['symbol'].tolist()
                return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []
    
    async def _screen_symbols_concurrent(self, 
                                       symbols: List[str],
                                       start_date: str,
                                       end_date: str,
                                       min_volume: int,
                                       min_price: float,
                                       max_workers: int,
                                       analysis_mode: str) -> List[Dict]:
        """Screen symbols concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._screen_single_symbol, 
                    symbol, start_date, end_date, min_volume, min_price, analysis_mode
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
                            start_date: str,
                            end_date: str,
                            min_volume: int,
                            min_price: float,
                            analysis_mode: str) -> Optional[Dict]:
        """Screen a single symbol using historical data."""
        try:
            # Get historical data from database
            historical_data = self._get_historical_data(symbol, start_date, end_date)
            
            if historical_data.empty or len(historical_data) < 30:
                return None
            
            # Get current price and volume from latest data
            current_price = historical_data['close_price'].iloc[-1]
            volume = historical_data['volume'].iloc[-1]
            avg_volume = historical_data['volume'].rolling(20).mean().iloc[-1]
            
            # Apply filters
            if avg_volume < min_volume or current_price < min_price:
                return None
            
            # Calculate indicators based on analysis mode
            indicators = self._calculate_indicators(historical_data, analysis_mode)
            
            # Generate signal based on analysis mode
            signal = self._generate_signal(indicators, historical_data, analysis_mode)
            
            if signal['signal'] == 'NEUTRAL':
                return None
            
            # Calculate levels
            levels = self._calculate_levels(historical_data, signal['signal'])
            
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
                'volume': volume,
                'avg_volume': avg_volume,
                'volume_ratio': indicators.get('volume_ratio', {}).iloc[-1] if 'volume_ratio' in indicators else 1.0,
                'volatility': levels.get('volatility', 0),
                'analysis_mode': analysis_mode,
                'screening_date': datetime.now().isoformat()
            }
            
            # Debug: Log first few signals
            if symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']:
                logger.info(f"🔍 Sample Signal - {symbol}:")
                logger.info(f"   Signal: {result['signal']}")
                logger.info(f"   Confidence: {result['confidence']}%")
                logger.info(f"   Risk-Reward: {result['risk_reward_ratio']}")
                logger.info(f"   Volume Ratio: {result['volume_ratio']}")
                logger.info(f"   Volatility: {result['volatility']}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error screening {symbol}: {e}")
            return None
    
    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data from database."""
        try:
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            with duckdb.connect(self.db_path) as conn:
                df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()  # Sort by date ascending
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame, analysis_mode: str) -> Dict:
        """Calculate technical indicators based on analysis mode."""
        indicators = {}
        
        if df.empty or len(df) < 20:
            return indicators
        
        # Basic indicators (all modes)
        indicators['sma_20'] = df['close_price'].rolling(20).mean()
        indicators['sma_50'] = df['close_price'].rolling(50).mean()
        
        # RSI
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume analysis
        indicators['volume_sma'] = df['volume'].rolling(20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        # Price action
        indicators['high_20'] = df['high_price'].rolling(20).max()
        indicators['low_20'] = df['low_price'].rolling(20).min()
        
        # Enhanced indicators (enhanced and comprehensive modes)
        if analysis_mode in ['enhanced', 'comprehensive']:
            # EMA
            indicators['ema_12'] = df['close_price'].ewm(span=12).mean()
            indicators['ema_26'] = df['close_price'].ewm(span=26).mean()
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            indicators['bb_middle'] = df['close_price'].rolling(20).mean()
            bb_std = df['close_price'].rolling(20).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
            indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
        
        return indicators
    
    def _generate_signal(self, indicators: Dict, df: pd.DataFrame, analysis_mode: str) -> Dict:
        """Generate trading signal based on analysis mode."""
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
        
        # RSI analysis
        if 30 < indicators['rsi'].iloc[-1] < 70:
            bullish_score += 1
            reasons.append("RSI in neutral zone")
        
        # Volume confirmation
        if indicators['volume_ratio'].iloc[-1] > 1.5:
            bullish_score += 1
            reasons.append("High volume confirmation")
        
        # Enhanced analysis for enhanced and comprehensive modes
        if analysis_mode in ['enhanced', 'comprehensive']:
            # MACD bullish
            if (indicators['macd'].iloc[-1] > indicators['macd_signal'].iloc[-1] and 
                indicators['macd_histogram'].iloc[-1] > indicators['macd_histogram'].iloc[-2]):
                bullish_score += 2
                reasons.append("MACD bullish crossover")
            
            # RSI oversold bounce
            if 30 < indicators['rsi'].iloc[-1] < 50:
                bullish_score += 1
                reasons.append("RSI oversold bounce")
            
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
        
        # RSI overbought
        if indicators['rsi'].iloc[-1] > 70:
            bearish_score += 1
        
        # Enhanced analysis for enhanced and comprehensive modes
        if analysis_mode in ['enhanced', 'comprehensive']:
            # MACD bearish
            if (indicators['macd'].iloc[-1] < indicators['macd_signal'].iloc[-1] and 
                indicators['macd_histogram'].iloc[-1] < indicators['macd_histogram'].iloc[-2]):
                bearish_score += 2
            
            # Breakdown from Bollinger Bands
            if current_price < indicators['bb_lower'].iloc[-1]:
                bearish_score += 2
        
        # Determine signal based on analysis mode
        if analysis_mode == 'basic':
            required_score = 3
            confidence_multiplier = 8
        elif analysis_mode == 'enhanced':
            required_score = 4
            confidence_multiplier = 7
        else:  # comprehensive
            required_score = 5
            confidence_multiplier = 6
        
        if bullish_score >= required_score and bullish_score > bearish_score:
            signal = 'BULLISH'
            # More realistic confidence calculation
            base_confidence = 40 + (bullish_score * confidence_multiplier)
            # Add volume bonus
            if indicators['volume_ratio'].iloc[-1] > 2.0:
                base_confidence += 10
            # Add trend strength bonus
            if current_price > indicators['sma_50'].iloc[-1]:
                base_confidence += 5
            confidence = min(85, base_confidence)
        elif bearish_score >= required_score and bearish_score > bullish_score:
            signal = 'BEARISH'
            # More realistic confidence calculation
            base_confidence = 40 + (bearish_score * confidence_multiplier)
            # Add volume bonus
            if indicators['volume_ratio'].iloc[-1] > 2.0:
                base_confidence += 10
            # Add trend strength bonus
            if current_price < indicators['sma_50'].iloc[-1]:
                base_confidence += 5
            confidence = min(85, base_confidence)
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
        
        # Calculate volatility-based multipliers
        volatility = df['close_price'].pct_change().std() * 100  # Volatility as percentage
        
        # Dynamic multipliers based on volatility
        if volatility < 2:  # Low volatility
            sl_multiplier = 1.5
            t1_multiplier = 2.5
            t2_multiplier = 4.0
            t3_multiplier = 6.0
        elif volatility < 4:  # Medium volatility
            sl_multiplier = 2.0
            t1_multiplier = 3.5
            t2_multiplier = 5.5
            t3_multiplier = 8.0
        else:  # High volatility
            sl_multiplier = 2.5
            t1_multiplier = 4.5
            t2_multiplier = 7.0
            t3_multiplier = 10.0
        
        if signal == 'BULLISH':
            entry = current_price
            stop_loss = entry - (atr * sl_multiplier)
            target1 = entry + (atr * t1_multiplier)
            target2 = entry + (atr * t2_multiplier)
            target3 = entry + (atr * t3_multiplier)
        else:  # BEARISH
            entry = current_price
            stop_loss = entry + (atr * sl_multiplier)
            target1 = entry - (atr * t1_multiplier)
            target2 = entry - (atr * t2_multiplier)
            target3 = entry - (atr * t3_multiplier)
        
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
            'risk_reward': round(risk_reward, 2),
            'volatility': round(volatility, 2)
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
            bullish_file = f"{self.results_dir}/unified_bullish_signals_{timestamp}.csv"
            bullish_df.to_csv(bullish_file, index=False)
            logger.info(f"📈 Bullish signals saved to {bullish_file}")
        
        # Save bearish signals
        if bearish_signals:
            bearish_df = pd.DataFrame(bearish_signals)
            bearish_file = f"{self.results_dir}/unified_bearish_signals_{timestamp}.csv"
            bearish_df.to_csv(bearish_file, index=False)
            logger.info(f"📉 Bearish signals saved to {bearish_file}")
        
        # Save summary
        summary_file = f"{self.results_dir}/unified_screening_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        logger.info(f"📊 Summary saved to {summary_file}")

# Global instance
unified_eod_screener = UnifiedEODScreener()
