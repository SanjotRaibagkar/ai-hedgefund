#!/usr/bin/env python3
"""
Intraday Data Collection Background Runner
Automatically collects intraday ML data every 15 minutes during market hours.
"""

import sys
import os
import time
import schedule
import subprocess
import psutil
import signal
from datetime import datetime, time as dt_time
from loguru import logger
import threading

# Add src to path
sys.path.append('./src')

from nsedata.NseUtility import NseUtils


class IntradayDataCollectorRunner:
    """Background runner for intraday data collection."""
    
    def __init__(self):
        """Initialize the background runner."""
        self.nse = NseUtils()
        self.data_collector_process = None
        self.is_running = False
        self.log_file = "logs/intraday_data_collection_background.log"
        
        # Setup logging
        logger.remove()
        logger.add(
            self.log_file,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
        )
        
        logger.info("🚀 Intraday Data Collection Background Runner initialized")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            current_time = datetime.now().time()
            market_start = dt_time(9, 30)  # 9:30 AM IST
            market_end = dt_time(15, 30)   # 3:30 PM IST
            
            # Check if current time is within market hours
            if market_start <= current_time <= market_end:
                # Check if it's a trading day (not a holiday)
                today = datetime.now().strftime('%d-%b-%Y')
                return not self.nse.is_nse_trading_holiday(today)
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking market status: {e}")
            return False
    
    def start_data_collector(self):
        """Start the intraday data collection process."""
        try:
            if self.data_collector_process and self.data_collector_process.poll() is None:
                logger.info("ℹ️ Intraday data collector is already running")
                return
            
            # Start the data collection script
            cmd = [
                sys.executable, 
                "src/intradayML/run_intraday_ml.py", 
                "--demo", "data"
            ]
            
            self.data_collector_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"🚀 Intraday data collector started with PID: {self.data_collector_process.pid}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start intraday data collector: {e}")
    
    def stop_data_collector(self):
        """Stop the intraday data collection process."""
        try:
            if self.data_collector_process and self.data_collector_process.poll() is None:
                # Send termination signal
                self.data_collector_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.data_collector_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    self.data_collector_process.kill()
                    self.data_collector_process.wait()
                
                logger.info("🛑 Intraday data collector stopped")
            else:
                logger.info("ℹ️ Intraday data collector is not running")
                
        except Exception as e:
            logger.error(f"❌ Error stopping intraday data collector: {e}")
    
    def run_data_collection(self):
        """Run data collection if market is open."""
        try:
            if self.is_market_open():
                logger.info("📊 Market is open - running data collection...")
                self.start_data_collector()
            else:
                logger.info("⏰ Market is closed - skipping data collection")
                
        except Exception as e:
            logger.error(f"❌ Error in data collection run: {e}")
    
    def schedule_daily_runs(self):
        """Schedule daily start and stop times."""
        try:
            # Schedule to start at 9:30 AM IST
            schedule.every().day.at("09:30").do(self.start_data_collector)
            
            # Schedule to stop at 3:30 PM IST
            schedule.every().day.at("15:30").do(self.stop_data_collector)
            
            # Schedule data collection every 15 minutes during market hours
            schedule.every(15).minutes.do(self.run_data_collection)
            
            logger.info("📅 Scheduled intraday data collection: Start at 09:30, Stop at 15:30")
            logger.info("⏰ Data collection interval: Every 15 minutes")
            
        except Exception as e:
            logger.error(f"❌ Error scheduling daily runs: {e}")
    
    def is_data_collector_running(self) -> bool:
        """Check if data collector process is running."""
        try:
            if self.data_collector_process:
                return self.data_collector_process.poll() is None
            return False
        except Exception:
            return False
    
    def run_background_scheduler(self):
        """Run the background scheduler."""
        try:
            logger.info("🔄 Starting background scheduler...")
            
            # Schedule daily runs
            self.schedule_daily_runs()
            
            self.is_running = True
            
            while self.is_running:
                try:
                    # Run pending scheduled tasks
                    schedule.run_pending()
                    
                    # Log status every minute
                    if self.is_data_collector_running():
                        logger.debug("✅ Intraday data collector is running")
                    else:
                        logger.debug("ℹ️ Intraday data collector is not running")
                    
                    # Sleep for 1 minute
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in scheduler loop: {e}")
                    time.sleep(60)  # Continue after error
            
            # Cleanup
            self.stop_data_collector()
            logger.info("🔚 Background scheduler stopped")
            
        except Exception as e:
            logger.error(f"❌ Fatal error in background scheduler: {e}")
    
    def stop(self):
        """Stop the background scheduler."""
        self.is_running = False
        self.stop_data_collector()


def main():
    """Main function."""
    print("🚀 Intraday Data Collection Background Runner")
    print("="*60)
    print("📅 Trading hours: 09:30 - 15:30 IST")
    print("📊 Auto-start/stop daily")
    print("⏰ Collection interval: Every 15 minutes")
    print("📝 Logs: logs/intraday_data_collection_background.log")
    print("="*60)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    try:
        # Initialize and run the background scheduler
        runner = IntradayDataCollectorRunner()
        runner.run_background_scheduler()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
