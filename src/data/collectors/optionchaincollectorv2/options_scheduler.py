#!/usr/bin/env python3
"""
Options Data Collection Scheduler
Automated scheduler for options data collection and batch processing.
"""

import schedule
import time
import logging
import os
import sys
from datetime import datetime, time as dt_time
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.collectors.optionchaincollectorv2.options_process_manager import OptionsProcessManager

class OptionsScheduler:
    """Scheduler for options data collection processes."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.logger = logging.getLogger(__name__)
        self.manager = OptionsProcessManager()
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        
        self.logger.info("🚀 Options Scheduler initialized")
    
    def _is_market_day(self) -> bool:
        """Check if today is a market day (Monday to Friday)."""
        return datetime.now().weekday() < 5
    
    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours."""
        now = datetime.now()
        current_time = now.time()
        
        # Market hours: 9:15 AM to 3:30 PM
        market_start = dt_time(9, 15, 0)
        market_end = dt_time(15, 30, 0)
        
        return market_start <= current_time <= market_end
    
    def start_options_processes(self):
        """Start options collection and batch processing processes."""
        if not self._is_market_day():
            self.logger.info("📅 Not a market day, skipping process start")
            return
        
        if not self._is_market_hours():
            self.logger.info("⏰ Outside market hours, skipping process start")
            return
        
        try:
            self.logger.info("🚀 Starting options processes...")
            results = self.manager.start_all_processes()
            
            if results['collection'] and results['batch']:
                self.logger.info("✅ Options processes started successfully")
                self.running = True
            else:
                self.logger.error("❌ Failed to start options processes")
                
        except Exception as e:
            self.logger.error(f"❌ Error starting options processes: {e}")
    
    def stop_options_processes(self):
        """Stop options collection and batch processing processes."""
        try:
            self.logger.info("🛑 Stopping options processes...")
            results = self.manager.stop_all_processes()
            
            if results['collection'] and results['batch']:
                self.logger.info("✅ Options processes stopped successfully")
                self.running = False
            else:
                self.logger.warning("⚠️ Some processes may not have stopped properly")
                
        except Exception as e:
            self.logger.error(f"❌ Error stopping options processes: {e}")
    
    def check_process_health(self):
        """Check the health of running processes."""
        if not self.running:
            return
        
        try:
            status = self.manager.get_process_status()
            
            # Log status
            collection_alive = status['collection_process']['alive']
            batch_alive = status['batch_process']['alive']
            
            self.logger.info(f"📊 Process Health Check:")
            self.logger.info(f"   Collection Process: {'🟢 Running' if collection_alive else '🔴 Stopped'}")
            self.logger.info(f"   Batch Process: {'🟢 Running' if batch_alive else '🔴 Stopped'}")
            
            # Restart if any process is dead
            if not collection_alive or not batch_alive:
                self.logger.warning("⚠️ Some processes are dead, restarting...")
                self.stop_options_processes()
                time.sleep(5)
                self.start_options_processes()
                
        except Exception as e:
            self.logger.error(f"❌ Error checking process health: {e}")
    
    def setup_schedule(self):
        """Setup the scheduling for options processes."""
        self.logger.info("📅 Setting up options process schedule...")
        
        # Start processes at market open (9:15 AM)
        schedule.every().monday.at("09:15").do(self.start_options_processes)
        schedule.every().tuesday.at("09:15").do(self.start_options_processes)
        schedule.every().wednesday.at("09:15").do(self.start_options_processes)
        schedule.every().thursday.at("09:15").do(self.start_options_processes)
        schedule.every().friday.at("09:15").do(self.start_options_processes)
        
        # Stop processes at market close (3:30 PM)
        schedule.every().monday.at("15:30").do(self.stop_options_processes)
        schedule.every().tuesday.at("15:30").do(self.stop_options_processes)
        schedule.every().wednesday.at("15:30").do(self.stop_options_processes)
        schedule.every().thursday.at("15:30").do(self.stop_options_processes)
        schedule.every().friday.at("15:30").do(self.stop_options_processes)
        
        # Health check every 5 minutes during market hours
        schedule.every(5).minutes.do(self.check_process_health)
        
        self.logger.info("✅ Schedule setup completed")
        self.logger.info("   Start time: 9:15 AM (Monday-Friday)")
        self.logger.info("   Stop time: 3:30 PM (Monday-Friday)")
        self.logger.info("   Health check: Every 5 minutes")
    
    def run_scheduler(self):
        """Run the scheduler loop."""
        self.logger.info("🚀 Starting options scheduler...")
        
        # Setup schedule
        self.setup_schedule()
        
        # If it's currently market hours, start processes immediately
        if self._is_market_day() and self._is_market_hours():
            self.logger.info("⏰ Currently market hours, starting processes immediately...")
            self.start_options_processes()
        
        # Run scheduler loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Scheduler stopped by user")
                self.stop_options_processes()
                break
                
            except Exception as e:
                self.logger.error(f"❌ Error in scheduler loop: {e}")
                time.sleep(30)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the scheduler and processes."""
        status = {
            'scheduler_running': True,
            'processes_running': self.running,
            'market_day': self._is_market_day(),
            'market_hours': self._is_market_hours(),
            'process_status': self.manager.get_process_status() if self.manager else None
        }
        
        return status


def main():
    """Main function for running the scheduler."""
    scheduler = OptionsScheduler()
    
    try:
        scheduler.run_scheduler()
    except Exception as e:
        print(f"❌ Error running scheduler: {e}")
        scheduler.stop_options_processes()


if __name__ == "__main__":
    main()
