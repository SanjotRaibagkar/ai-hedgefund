#!/usr/bin/env python3
"""
Comprehensive Screening of All Stocks from Database
Run screening on all available stocks and save results to CSV
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

async def comprehensive_screening_all_stocks():
    """Run comprehensive screening on all stocks from database."""
    logger.info("🚀 Starting Comprehensive Screening of All Stocks")
    logger.info("=" * 60)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Get all available symbols
        logger.info("📊 Getting all available symbols from database...")
        all_symbols = db_manager.get_available_symbols()
        logger.info(f"✅ Found {len(all_symbols)} symbols in database")
        
        # Run comprehensive screening
        logger.info("🔍 Running comprehensive screening...")
        results = await unified_eod_screener.screen_universe(
            symbols=all_symbols,
            min_volume=50000,  # Lower threshold to get more signals
            min_price=5.0,     # Lower threshold to get more signals
            analysis_mode="comprehensive"
        )
        
        # Save detailed results to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save bullish signals
        if results['bullish_signals']:
            bullish_df = pd.DataFrame(results['bullish_signals'])
            bullish_file = f"results/comprehensive_bullish_signals_{timestamp}.csv"
            os.makedirs('results', exist_ok=True)
            bullish_df.to_csv(bullish_file, index=False)
            logger.info(f"📈 Bullish signals saved to: {bullish_file}")
            logger.info(f"   Count: {len(results['bullish_signals'])}")
        
        # Save bearish signals
        if results['bearish_signals']:
            bearish_df = pd.DataFrame(results['bearish_signals'])
            bearish_file = f"results/comprehensive_bearish_signals_{timestamp}.csv"
            bearish_df.to_csv(bearish_file, index=False)
            logger.info(f"📉 Bearish signals saved to: {bearish_file}")
            logger.info(f"   Count: {len(results['bearish_signals'])}")
        
        # Create summary report
        summary = results['summary']
        logger.info("\n📊 COMPREHENSIVE SCREENING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Symbols Available: {len(all_symbols)}")
        logger.info(f"Total Symbols Screened: {summary['total_screened']}")
        logger.info(f"Bullish Signals: {summary['bullish_signals']}")
        logger.info(f"Bearish Signals: {summary['bearish_signals']}")
        logger.info(f"Total Signals: {summary['bullish_signals'] + summary['bearish_signals']}")
        
        # Show top signals if any
        if results['bullish_signals']:
            logger.info("\n📈 TOP 5 BULLISH SIGNALS:")
            logger.info("-" * 40)
            for i, signal in enumerate(results['bullish_signals'][:5], 1):
                logger.info(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence")
                logger.info(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}")
                logger.info(f"   Risk-Reward: {signal['risk_reward_ratio']:.2f}")
                logger.info(f"   Reasons: {', '.join(signal['reasons'][:3])}")
                logger.info("")
        
        if results['bearish_signals']:
            logger.info("\n📉 TOP 5 BEARISH SIGNALS:")
            logger.info("-" * 40)
            for i, signal in enumerate(results['bearish_signals'][:5], 1):
                logger.info(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence")
                logger.info(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}")
                logger.info(f"   Risk-Reward: {signal['risk_reward_ratio']:.2f}")
                logger.info(f"   Reasons: {', '.join(signal['reasons'][:3])}")
                logger.info("")
        
        # Save summary to file
        summary_file = f"results/comprehensive_screening_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE SCREENING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Symbols Available: {len(all_symbols)}\n")
            f.write(f"Total Symbols Screened: {summary['total_screened']}\n")
            f.write(f"Bullish Signals: {summary['bullish_signals']}\n")
            f.write(f"Bearish Signals: {summary['bearish_signals']}\n")
            f.write(f"Total Signals: {summary['bullish_signals'] + summary['bearish_signals']}\n\n")
            
            if results['bullish_signals']:
                f.write("TOP 10 BULLISH SIGNALS:\n")
                f.write("-" * 30 + "\n")
                for i, signal in enumerate(results['bullish_signals'][:10], 1):
                    f.write(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence\n")
                    f.write(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}\n")
                    f.write(f"   Risk-Reward: {signal['risk_reward_ratio']:.2f}\n")
                    f.write(f"   Reasons: {', '.join(signal['reasons'])}\n\n")
            
            if results['bearish_signals']:
                f.write("TOP 10 BEARISH SIGNALS:\n")
                f.write("-" * 30 + "\n")
                for i, signal in enumerate(results['bearish_signals'][:10], 1):
                    f.write(f"{i}. {signal['symbol']} - {signal['confidence']}% confidence\n")
                    f.write(f"   Entry: ₹{signal['entry_price']} | SL: ₹{signal['stop_loss']} | T1: ₹{signal['targets']['T1']}\n")
                    f.write(f"   Risk-Reward: {signal['risk_reward_ratio']:.2f}\n")
                    f.write(f"   Reasons: {', '.join(signal['reasons'])}\n\n")
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in comprehensive screening: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(comprehensive_screening_all_stocks())
