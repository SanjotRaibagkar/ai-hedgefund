#!/usr/bin/env python3
"""
Background Options Analysis Scheduler Runner
Starts the options analysis scheduler at 9:30 AM and stops at 3:30 PM IST daily.
"""

import sys
import os
import time
import schedule
import subprocess
import signal
import psutil
from datetime import datetime, timedelta, date
from loguru import logger
import sys
import os

# Add src to path
sys.path.append('./src')

from nsedata.NseUtility import NseUtils


class OptionsAnalysisBackgroundRunner:
    """Background runner for options analysis scheduler."""
    
    def __init__(self):
        """Initialize the background runner."""
        self.process = None
        self.nse = NseUtils()
        self.market_open = "09:30"
        self.market_close = "15:30"
        
        # Setup logging
        logger.add("logs/options_analysis_background.log", rotation="1 day", retention="7 days")
        
        logger.info("🚀 Options Analysis Background Runner initialized")
        
    def _get_trading_holidays(self):
        """Get trading holidays from NSE."""
        try:
            return self.nse.trading_holidays(list_only=True)
        except Exception as e:
            logger.error(f"❌ Error fetching trading holidays: {e}")
            return []
    
    def _is_trading_day(self):
        """Check if today is a trading day."""
        try:
            today = date.today()
            trading_holidays = self._get_trading_holidays()
            
            # Check if today is weekend
            if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if today is a trading holiday
            today_str = today.strftime('%Y-%m-%d')
            if today_str in trading_holidays:
                return False
                
            return True
        except Exception as e:
            logger.error(f"❌ Error checking trading day: {e}")
            return False
    
    def start_analysis_scheduler(self):
        """Start the options analysis scheduler process."""
        try:
            if self.process and self.process.poll() is None:
                logger.info("✅ Options analysis scheduler is already running")
                return
            
            logger.info("🚀 Starting options analysis scheduler...")
            
            # Start the analysis scheduler process
            self.process = subprocess.Popen([
                sys.executable, 
                "junk/run_options_scheduler.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"✅ Options analysis scheduler started with PID: {self.process.pid}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start options analysis scheduler: {e}")
    
    def stop_analysis_scheduler(self):
        """Stop the options analysis scheduler process."""
        try:
            if self.process and self.process.poll() is None:
                logger.info("🛑 Stopping options analysis scheduler...")
                
                # Send SIGTERM signal
                self.process.terminate()
                
                # Wait for graceful shutdown (5 seconds)
                try:
                    self.process.wait(timeout=5)
                    logger.info("✅ Options analysis scheduler stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    logger.warning("⚠️ Force killing options analysis scheduler...")
                    self.process.kill()
                    self.process.wait()
                    logger.info("✅ Options analysis scheduler force stopped")
                
                self.process = None
            else:
                logger.info("ℹ️ Options analysis scheduler is not running")
                
        except Exception as e:
            logger.error(f"❌ Failed to stop options analysis scheduler: {e}")
    
    def check_scheduler_status(self):
        """Check if the analysis scheduler process is running."""
        try:
            if self.process and self.process.poll() is None:
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error checking scheduler status: {e}")
            return False
    
    def schedule_daily_runs(self):
        """Schedule daily start and stop times."""
        try:
            # Schedule start at 9:30 AM
            schedule.every().day.at(self.market_open).do(self.start_analysis_scheduler)
            
            # Schedule stop at 3:30 PM
            schedule.every().day.at(self.market_close).do(self.stop_analysis_scheduler)
            
            logger.info(f"📅 Scheduled options analysis: Start at {self.market_open}, Stop at {self.market_close}")
            
        except Exception as e:
            logger.error(f"❌ Failed to schedule daily runs: {e}")
    
    def run_background_scheduler(self):
        """Run the background scheduler."""
        try:
            logger.info("🔄 Starting background scheduler...")
            
            # Schedule daily runs
            self.schedule_daily_runs()
            
            # Check if we should start now
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            
            if self._is_trading_day() and self.market_open <= current_time <= self.market_close:
                logger.info("📊 Market is open, starting analysis scheduler now...")
                self.start_analysis_scheduler()
            
            # Main scheduler loop
            while True:
                try:
                    schedule.run_pending()
                    
                    # Check scheduler status every 5 minutes
                    if self.check_scheduler_status():
                        logger.debug("✅ Options analysis scheduler is running")
                    else:
                        logger.debug("ℹ️ Options analysis scheduler is not running")
                    
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Background scheduler stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Scheduler error: {e}")
                    time.sleep(60)
            
            # Cleanup on exit
            self.stop_analysis_scheduler()
            
        except Exception as e:
            logger.error(f"❌ Fatal error in background scheduler: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.stop_analysis_scheduler()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def main():
    """Main function."""
    print("🚀 Options Analysis Scheduler Background Runner")
    print("=" * 60)
    print(f"📅 Trading hours: 09:30 - 15:30 IST")
    print(f"📊 Auto-start/stop daily")
    print(f"⏰ Analysis interval: Every 15 minutes")
    print(f"📝 Logs: logs/options_analysis_background.log")
    print("=" * 60)
    
    runner = None
    try:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        runner = OptionsAnalysisBackgroundRunner()
        runner.run_background_scheduler()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping background runner...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if runner:
            runner.cleanup()
        print("👋 Background runner stopped")


if __name__ == "__main__":
    main()
