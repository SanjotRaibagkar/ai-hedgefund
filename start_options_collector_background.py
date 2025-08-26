#!/usr/bin/env python3
"""
Background Options Chain Collector Runner
Starts the options chain collector at 9:30 AM and stops at 3:30 PM IST daily.
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


class OptionsCollectorBackgroundRunner:
    """Background runner for options chain collector."""
    
    def __init__(self):
        """Initialize the background runner."""
        self.process = None
        self.nse = NseUtils()
        self.market_open = "09:30"
        self.market_close = "15:30"
        
        # Setup logging
        logger.add("logs/options_collector_background.log", rotation="1 day", retention="7 days")
        
        logger.info("🚀 Options Collector Background Runner initialized")
        
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
    
    def start_collector(self):
        """Start the options chain collector process."""
        try:
            if self.process and self.process.poll() is None:
                logger.info("✅ Options collector is already running")
                return
            
            logger.info("🚀 Starting options chain collector...")
            
            # Start the collector process
            self.process = subprocess.Popen([
                sys.executable, 
                "src/data/downloaders/options_chain_collector.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"✅ Options collector started with PID: {self.process.pid}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start options collector: {e}")
    
    def stop_collector(self):
        """Stop the options chain collector process."""
        try:
            if self.process and self.process.poll() is None:
                logger.info("🛑 Stopping options chain collector...")
                
                # Send SIGTERM signal
                self.process.terminate()
                
                # Wait for graceful shutdown (5 seconds)
                try:
                    self.process.wait(timeout=5)
                    logger.info("✅ Options collector stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    logger.warning("⚠️ Force killing options collector...")
                    self.process.kill()
                    self.process.wait()
                    logger.info("✅ Options collector force stopped")
                
                self.process = None
            else:
                logger.info("ℹ️ Options collector is not running")
                
        except Exception as e:
            logger.error(f"❌ Failed to stop options collector: {e}")
    
    def check_collector_status(self):
        """Check if the collector process is running."""
        try:
            if self.process and self.process.poll() is None:
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error checking collector status: {e}")
            return False
    
    def schedule_daily_runs(self):
        """Schedule daily start and stop times."""
        try:
            # Schedule start at 9:30 AM
            schedule.every().day.at(self.market_open).do(self.start_collector)
            
            # Schedule stop at 3:30 PM
            schedule.every().day.at(self.market_close).do(self.stop_collector)
            
            logger.info(f"📅 Scheduled options collector: Start at {self.market_open}, Stop at {self.market_close}")
            
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
                logger.info("📊 Market is open, starting collector now...")
                self.start_collector()
            
            # Main scheduler loop
            while True:
                try:
                    schedule.run_pending()
                    
                    # Check collector status every minute
                    if self.check_collector_status():
                        logger.debug("✅ Options collector is running")
                    else:
                        logger.debug("ℹ️ Options collector is not running")
                        # Auto-restart if not running and market is open
                        if self._is_trading_day() and self.market_open <= datetime.now().strftime('%H:%M') <= self.market_close:
                            logger.info("🔄 Auto-restarting options collector...")
                            self.start_collector()
                    
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Background scheduler stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Scheduler error: {e}")
                    time.sleep(60)
            
            # Cleanup on exit
            self.stop_collector()
            
        except Exception as e:
            logger.error(f"❌ Fatal error in background scheduler: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.stop_collector()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def main():
    """Main function."""
    print("🚀 Options Chain Collector Background Runner")
    print("=" * 60)
    print(f"📅 Trading hours: 09:30 - 15:30 IST")
    print(f"📊 Auto-start/stop daily")
    print(f"📝 Logs: logs/options_collector_background.log")
    print("=" * 60)
    
    runner = None
    try:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        runner = OptionsCollectorBackgroundRunner()
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
