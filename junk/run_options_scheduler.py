#!/usr/bin/env python3
"""
Run Options Scheduler
Simple script to run options analysis every 15 minutes using unified EnhancedOptionsAnalyzer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import schedule
from datetime import datetime
from loguru import logger

from src.screening.enhanced_options_analyzer import EnhancedOptionsAnalyzer

def run_options_analysis(index: str = 'NIFTY'):
    """Run options analysis for given index using unified analyzer."""
    logger.info(f"🎯 Running {index} options analysis at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Use unified options analyzer
        analyzer = EnhancedOptionsAnalyzer()
        
        # Run analysis and save to CSV with performance tracking
        success = analyzer.run_analysis_and_save(index)
        
        if success:
            # Get the result for logging
            result = analyzer.analyze_index_options(index)
            if result:
                logger.info(f"📊 {index} Analysis Results:")
                logger.info(f"   Spot Price: ₹{result['current_price']:,.0f}")
                logger.info(f"   ATM Strike: ₹{result['atm_strike']:,.0f}")
                logger.info(f"   PCR: {result['oi_analysis']['pcr']:.2f}")
                logger.info(f"   Signal: {result['signal']['signal']} (Confidence: {result['signal']['confidence']:.1f}%)")
                logger.info(f"   Trade: {result['signal']['suggested_trade']}")
                logger.info(f"📊 Record saved to {analyzer.csv_file}")
            else:
                logger.error(f"❌ Failed to get {index} analysis results")
        else:
            logger.error(f"❌ Failed to run {index} analysis and save")
            
    except Exception as e:
        logger.error(f"❌ Error in {index} analysis: {e}")

def start_scheduler():
    """Start the options analysis scheduler."""
    logger.info("🚀 Starting Options Scheduler (every 15 minutes)")
    
    # Schedule NIFTY analysis every 15 minutes
    schedule.every(15).minutes.do(run_options_analysis, 'NIFTY')
    
    # Schedule BANKNIFTY analysis every 15 minutes (with 5 second delay)
    schedule.every(15).minutes.do(lambda: time.sleep(5) or run_options_analysis('BANKNIFTY'))
    
    # Run initial analysis
    logger.info("📊 Running initial analysis...")
    run_options_analysis('NIFTY')
    time.sleep(5)
    run_options_analysis('BANKNIFTY')
    
    # Keep scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    start_scheduler()
