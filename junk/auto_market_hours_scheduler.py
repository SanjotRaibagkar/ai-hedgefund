#!/usr/bin/env python3
"""
Auto Market Hours Options Scheduler
Automatically starts at 9:30 AM and stops at 3:30 PM on trading days.
Runs options analysis every 15 minutes during market hours only.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import schedule
import threading
from datetime import datetime, timedelta
from loguru import logger

from src.screening.fixed_enhanced_options_analyzer import FixedEnhancedOptionsAnalyzer
from src.nsedata.NseUtility import NseUtils


class AutoMarketHoursScheduler:
    """Scheduler that automatically runs during market hours on trading days."""
    
    def __init__(self):
        """Initialize the auto market hours scheduler."""
        self.logger = logger
        self.nse = NseUtils()
        self.running = False
        self.scheduler_thread = None
        
        # Market hours (IST)
        self.market_open = "09:30"
        self.market_close = "15:30"
        
        # Analysis interval
        self.analysis_interval = 15  # minutes
        
        # Trading holidays cache
        self.trading_holidays = None
        self.last_holiday_check = None
        
        self.logger.info("🚀 Auto Market Hours Scheduler initialized")
        self.logger.info(f"📅 Market hours: {self.market_open} - {self.market_close} IST")
        self.logger.info(f"⏰ Analysis interval: {self.analysis_interval} minutes")
    
    def _is_trading_day(self) -> bool:
        """Check if today is a trading day."""
        try:
            today = datetime.now().date()
            
            # Check if it's weekend
            if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check trading holidays (cache for 1 day)
            if (self.last_holiday_check is None or 
                self.last_holiday_check.date() != today):
                
                self.trading_holidays = self.nse.trading_holidays()
                self.last_holiday_check = datetime.now()
                
                self.logger.info(f"📅 Updated trading holidays cache")
            
            if self.trading_holidays is not None:
                try:
                    holiday_dates = [holiday['date'] for holiday in self.trading_holidays]
                    if today.strftime('%Y-%m-%d') in holiday_dates:
                        self.logger.info(f"📅 Today is a trading holiday")
                        return False
                except (KeyError, TypeError):
                    # If holiday data format is unexpected, just log and continue
                    self.logger.warning(f"⚠️ Could not parse trading holidays, assuming trading day")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking trading day: {e}")
            # Default to True if we can't check
            return True
    
    def _is_market_hours(self) -> bool:
        """Check if it's currently market hours."""
        try:
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            
            # Check if current time is between market open and close
            is_market_hours = self.market_open <= current_time <= self.market_close
            
            if is_market_hours:
                self.logger.debug(f"⏰ Market hours: {current_time} (Open: {self.market_open}, Close: {self.market_close})")
            else:
                self.logger.debug(f"⏰ Outside market hours: {current_time}")
            
            return is_market_hours
            
        except Exception as e:
            self.logger.error(f"❌ Error checking market hours: {e}")
            return False
    
    def _should_run_analysis(self) -> bool:
        """Check if analysis should run (trading day + market hours)."""
        is_trading_day = self._is_trading_day()
        is_market_hours = self._is_market_hours()
        
        should_run = is_trading_day and is_market_hours
        
        if should_run:
            self.logger.debug(f"✅ Conditions met: Trading day={is_trading_day}, Market hours={is_market_hours}")
        else:
            self.logger.debug(f"❌ Conditions not met: Trading day={is_trading_day}, Market hours={is_market_hours}")
        
        return should_run
    
    def run_options_analysis(self, index: str = 'NIFTY'):
        """Run options analysis for given index if conditions are met."""
        if not self._should_run_analysis():
            self.logger.info(f"⏸️ Skipping {index} analysis - outside market hours or non-trading day")
            return
        
        self.logger.info(f"🎯 Running {index} options analysis at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Use fixed enhanced options analyzer
            analyzer = FixedEnhancedOptionsAnalyzer()
            
            # Run analysis and save to CSV with performance tracking
            success = analyzer.run_analysis_and_save(index)
            
            if success:
                # Get the result for logging
                result = analyzer.analyze_index_options(index)
                if result:
                    self.logger.info(f"📊 {index} Analysis Results:")
                    self.logger.info(f"   Spot Price: ₹{result['current_price']:,.0f}")
                    self.logger.info(f"   ATM Strike: ₹{result['atm_strike']:,.0f}")
                    self.logger.info(f"   PCR: {result['oi_analysis']['pcr']:.2f}")
                    self.logger.info(f"   Signal: {result['signal']['signal']} (Confidence: {result['signal']['confidence']:.1f}%)")
                    self.logger.info(f"   Trade: {result['signal']['suggested_trade']}")
                    self.logger.info(f"📊 Record saved to {analyzer.csv_file}")
                else:
                    self.logger.error(f"❌ Failed to get {index} analysis results")
            else:
                self.logger.error(f"❌ Failed to run {index} analysis and save")
                
        except Exception as e:
            self.logger.error(f"❌ Error in {index} analysis: {e}")
    
    def start_market_analysis(self):
        """Start analysis when market opens."""
        self.logger.info("🚀 Market opened - starting options analysis")
        self.running = True
        
        # Run initial analysis
        self.logger.info("📊 Running initial market analysis...")
        self.run_options_analysis('NIFTY')
        time.sleep(5)
        self.run_options_analysis('BANKNIFTY')
    
    def stop_market_analysis(self):
        """Stop analysis when market closes."""
        self.logger.info("🛑 Market closed - stopping options analysis")
        self.running = False
    
    def schedule_market_analysis(self):
        """Schedule analysis during market hours."""
        # Schedule NIFTY analysis every 15 minutes during market hours
        schedule.every(self.analysis_interval).minutes.do(self.run_options_analysis, 'NIFTY')
        
        # Schedule BANKNIFTY analysis every 15 minutes (with 5 second delay)
        schedule.every(self.analysis_interval).minutes.do(
            lambda: time.sleep(5) or self.run_options_analysis('BANKNIFTY')
        )
        
        self.logger.info(f"⏰ Scheduled analysis every {self.analysis_interval} minutes during market hours")
    
    def start_scheduler(self):
        """Start the auto market hours scheduler."""
        self.logger.info("🚀 Starting Auto Market Hours Options Scheduler")
        
        # Schedule market open and close
        schedule.every().day.at(self.market_open).do(self.start_market_analysis)
        schedule.every().day.at(self.market_close).do(self.stop_market_analysis)
        
        # Schedule analysis during market hours
        self.schedule_market_analysis()
        
        # Check if we should start immediately
        if self._should_run_analysis():
            self.logger.info("⏰ Currently market hours, starting analysis immediately")
            self.start_market_analysis()
        else:
            self.logger.info("⏰ Outside market hours, waiting for market open...")
        
        # Keep scheduler running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                self.logger.info("🛑 Scheduler stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Scheduler error: {e}")
                time.sleep(60)
    
    def run_continuous(self):
        """Run analysis continuously (for testing)."""
        self.logger.info("🚀 Running continuous analysis (ignoring market hours)")
        
        # Schedule analysis every 15 minutes
        schedule.every(self.analysis_interval).minutes.do(self.run_options_analysis, 'NIFTY')
        schedule.every(self.analysis_interval).minutes.do(
            lambda: time.sleep(5) or self.run_options_analysis('BANKNIFTY')
        )
        
        # Run initial analysis
        self.logger.info("📊 Running initial analysis...")
        self.run_options_analysis('NIFTY')
        time.sleep(5)
        self.run_options_analysis('BANKNIFTY')
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                self.logger.info("🛑 Stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error: {e}")
                time.sleep(60)


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto Market Hours Options Scheduler')
    parser.add_argument('--mode', choices=['auto', 'continuous'], default='auto',
                       help='Scheduler mode: auto (respects market hours) or continuous (ignores market hours)')
    parser.add_argument('--interval', type=int, default=15,
                       help='Analysis interval in minutes (default: 15)')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/auto_options_scheduler.log", rotation="1 day", retention="7 days", level="DEBUG")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize scheduler
    scheduler = AutoMarketHoursScheduler()
    scheduler.analysis_interval = args.interval
    
    try:
        if args.mode == 'auto':
            logger.info("🚀 Starting in AUTO mode (respects market hours)")
            scheduler.start_scheduler()
        else:
            logger.info("🚀 Starting in CONTINUOUS mode (ignores market hours)")
            scheduler.run_continuous()
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()
