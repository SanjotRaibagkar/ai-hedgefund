#!/usr/bin/env python3
"""
Investigate Specific Stocks - Why No Signals Generated
Analyze why RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK are not generating signals
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from src.screening.unified_eod_screener import unified_eod_screener
from src.data.database.duckdb_manager import DatabaseManager

async def investigate_specific_stocks():
    """Investigate why specific stocks are not generating signals."""
    logger.info("🔍 Investigating Specific Stocks - Why No Signals")
    logger.info("=" * 60)
    
    # Test stocks
    test_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        for symbol in test_stocks:
            logger.info(f"\n📊 Analyzing {symbol}...")
            logger.info("-" * 40)
            
            # Get historical data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            df = db_manager.get_price_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning(f"❌ No data found for {symbol}")
                continue
            
            logger.info(f"✅ Data found: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
            
            # Get latest data
            latest = df.iloc[-1]
            logger.info(f"📈 Latest Price: ₹{latest['close_price']:.2f}")
            logger.info(f"📊 Volume: {latest['volume']:,}")
            
            # Check if meets basic criteria
            if latest['volume'] < 50000:
                logger.warning(f"⚠️ Volume too low: {latest['volume']:,} < 50,000")
            
            if latest['close_price'] < 5.0:
                logger.warning(f"⚠️ Price too low: ₹{latest['close_price']:.2f} < ₹5.0")
            
            # Calculate indicators manually to see what's happening
            screener = unified_eod_screener
            indicators = screener._calculate_indicators(df, "basic")
            
            # Check signal generation
            signal_result = screener._generate_signal(indicators, df, "basic")
            
            logger.info(f"🎯 Signal Analysis for {symbol}:")
            logger.info(f"   Signal: {signal_result['signal']}")
            logger.info(f"   Confidence: {signal_result['confidence']}%")
            logger.info(f"   Bullish Score: {signal_result['bullish_score']}")
            logger.info(f"   Bearish Score: {signal_result['bearish_score']}")
            logger.info(f"   Reasons: {', '.join(signal_result['reasons'])}")
            
            # Check individual indicators
            logger.info(f"📊 Technical Indicators for {symbol}:")
            if 'sma_20' in indicators and not indicators['sma_20'].empty:
                logger.info(f"   SMA 20: ₹{indicators['sma_20'].iloc[-1]:.2f}")
                logger.info(f"   Price vs SMA 20: {'Above' if latest['close_price'] > indicators['sma_20'].iloc[-1] else 'Below'}")
            
            if 'sma_50' in indicators and not indicators['sma_50'].empty:
                logger.info(f"   SMA 50: ₹{indicators['sma_50'].iloc[-1]:.2f}")
                logger.info(f"   Price vs SMA 50: {'Above' if latest['close_price'] > indicators['sma_50'].iloc[-1] else 'Below'}")
            
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi_value = indicators['rsi'].iloc[-1]
                logger.info(f"   RSI: {rsi_value:.2f}")
                if rsi_value < 30:
                    logger.info(f"   RSI Status: Oversold (< 30)")
                elif rsi_value > 70:
                    logger.info(f"   RSI Status: Overbought (> 70)")
                else:
                    logger.info(f"   RSI Status: Neutral (30-70)")
            
            if 'volume_ratio' in indicators and not indicators['volume_ratio'].empty:
                vol_ratio = indicators['volume_ratio'].iloc[-1]
                logger.info(f"   Volume Ratio: {vol_ratio:.2f}")
                logger.info(f"   Volume Status: {'High' if vol_ratio > 1.5 else 'Normal' if vol_ratio > 1.0 else 'Low'}")
            
            # Check what's missing for signal generation
            required_score = 3  # For basic mode
            current_score = max(signal_result['bullish_score'], signal_result['bearish_score'])
            
            if current_score < required_score:
                logger.warning(f"⚠️ Score too low: {current_score} < {required_score} (required for basic mode)")
                logger.info(f"   Need {required_score - current_score} more positive indicators")
            
            logger.info("")
        
        # Now test with different thresholds
        logger.info("\n🔧 Testing with Lower Thresholds...")
        logger.info("=" * 50)
        
        # Test with very low thresholds
        results = await unified_eod_screener.screen_universe(
            symbols=test_stocks,
            min_volume=1000,   # Very low volume threshold
            min_price=1.0,     # Very low price threshold
            analysis_mode="basic"
        )
        
        logger.info(f"📊 Results with low thresholds:")
        logger.info(f"   Total screened: {results['summary']['total_screened']}")
        logger.info(f"   Bullish signals: {results['summary']['bullish_signals']}")
        logger.info(f"   Bearish signals: {results['summary']['bearish_signals']}")
        
        if results['bullish_signals']:
            logger.info("📈 Bullish signals found:")
            for signal in results['bullish_signals']:
                logger.info(f"   {signal['symbol']} - {signal['confidence']}% confidence")
        
        if results['bearish_signals']:
            logger.info("📉 Bearish signals found:")
            for signal in results['bearish_signals']:
                logger.info(f"   {signal['symbol']} - {signal['confidence']}% confidence")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in investigation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(investigate_specific_stocks())
