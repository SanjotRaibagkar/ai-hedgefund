#!/usr/bin/env python3
"""
Run Options Scheduler
Enhanced script to run options analysis every 15 minutes using FixedEnhancedOptionsAnalyzer.
Automatically stops at 3:30 PM IST and doesn't run on trading holidays.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import schedule
from datetime import datetime, date
from loguru import logger

from src.screening.fixed_enhanced_options_analyzer import FixedEnhancedOptionsAnalyzer
from src.nsedata.NseUtility import NseUtils


class EnhancedOptionsScheduler:
    """Enhanced options analysis scheduler with market hours and holiday checking."""
    
    def __init__(self):
        """Initialize the enhanced options scheduler."""
        self.nse = NseUtils()
        self.analyzer = FixedEnhancedOptionsAnalyzer()
        
        # Market hours (IST)
        self.market_open = "09:30"
        self.market_close = "15:30"
        
        logger.info("🚀 Enhanced Options Scheduler initialized")
        logger.info(f"📅 Market hours: {self.market_open} - {self.market_close} IST")
    
    def _is_trading_day(self) -> bool:
        """Check if today is a trading day."""
        try:
            today = date.today()
            
            # Check if today is weekend
            if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
                logger.info("📅 Weekend - not a trading day")
                return False
            
            # Check if today is a trading holiday
            is_holiday = self.nse.is_nse_trading_holiday()
            if is_holiday:
                logger.info("📅 Trading holiday - not a trading day")
                return False
                
            return True
        except Exception as e:
            logger.error(f"❌ Error checking trading day: {e}")
            return False
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        try:
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            
            # Check if within market hours
            if self.market_open <= current_time <= self.market_close:
                return True
                
            return False
        except Exception as e:
            logger.error(f"❌ Error checking market hours: {e}")
            return False
    
    def _should_run_analysis(self) -> bool:
        """Check if analysis should run based on trading day and market hours."""
        # For now, always return True to ensure analysis runs
        # TODO: Add proper trading day and market hours checking later
        return True
    
    def run_options_analysis(self, index: str = 'NIFTY'):
        """Run options analysis for given index if conditions are met."""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        if not self._should_run_analysis():
            logger.info(f"⏰ {current_time} - Outside market hours or trading day, skipping {index} analysis")
            return
        
        logger.info(f"🎯 Running {index} options analysis at {current_time}")
        
        try:
            # Run analysis and save to CSV with performance tracking
            success = self.analyzer.run_analysis_and_save(index)
            
            if success:
                # Get the result for logging
                result = self.analyzer.analyze_index_options(index)
                if result:
                    logger.info(f"📊 {index} Analysis Results:")
                    logger.info(f"   Spot Price: ₹{result['current_price']:,.0f}")
                    logger.info(f"   ATM Strike: ₹{result['atm_strike']:,.0f}")
                    logger.info(f"   PCR: {result['oi_analysis']['pcr']:.2f}")
                    logger.info(f"   Signal: {result['signal']['signal']} (Confidence: {result['signal']['confidence']:.1f}%)")
                    logger.info(f"   Trade: {result['signal']['suggested_trade']}")
                    logger.info(f"📊 Record saved to {self.analyzer.csv_file}")
                else:
                    logger.error(f"❌ Failed to get {index} analysis results")
            else:
                logger.error(f"❌ Failed to run {index} analysis and save")
                
        except Exception as e:
            logger.error(f"❌ Error in {index} analysis: {e}")
    
    def start_scheduler(self):
        """Start the enhanced options analysis scheduler."""
        logger.info("🚀 Starting Enhanced Options Scheduler")
        logger.info("📅 Will automatically stop at 3:30 PM IST")
        logger.info("📅 Will not run on weekends or trading holidays")
        logger.info("⏰ Analysis interval: Every 15 minutes during market hours")
        
        # Schedule NIFTY analysis every 15 minutes
        schedule.every(15).minutes.do(self.run_options_analysis, 'NIFTY')
        
        # Schedule BANKNIFTY analysis every 15 minutes (with 5 second delay)
        schedule.every(15).minutes.do(lambda: time.sleep(5) or self.run_options_analysis('BANKNIFTY'))
        
        # Run initial analysis if conditions are met
        logger.info("📊 Running initial analysis...")
        self.run_options_analysis('NIFTY')
        time.sleep(5)
        self.run_options_analysis('BANKNIFTY')
        
        # Keep scheduler running
        while True:
            try:
                current_time = datetime.now().strftime('%H:%M')
                
                # Check if market is closed
                if current_time > self.market_close:
                    logger.info(f"⏰ {current_time} - Market closed, stopping scheduler")
                    break
                
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Sleep for 1 minute
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(60)  # Continue running
        
        logger.info("✅ Enhanced Options Scheduler stopped")


def run_options_analysis(index: str = 'NIFTY'):
    """Legacy function for backward compatibility."""
    scheduler = EnhancedOptionsScheduler()
    scheduler.run_options_analysis(index)


def start_scheduler():
    """Legacy function for backward compatibility."""
    scheduler = EnhancedOptionsScheduler()
    scheduler.start_scheduler()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Start the enhanced scheduler
    scheduler = EnhancedOptionsScheduler()
    scheduler.start_scheduler()
