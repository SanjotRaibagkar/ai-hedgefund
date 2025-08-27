#!/usr/bin/env python3
"""
Fundamental Data Collection Scheduler
Automated scheduler for downloading fundamental data from NSE.
"""

import schedule
import time
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.fundamental.collectors.nse_fundamental_collector import NSEFundamentalCollector

class FundamentalScheduler:
    """Scheduler for fundamental data collection."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.logger = logging.getLogger(__name__)
        self.collector = NSEFundamentalCollector()
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        
        self.logger.info("🚀 Fundamental Scheduler initialized")
    
    def _is_market_day(self) -> bool:
        """Check if today is a market day (Monday to Friday)."""
        return datetime.now().weekday() < 5
    
    def _should_run_download(self) -> bool:
        """Check if fundamental download should run."""
        # Only run on market days
        if not self._is_market_day():
            self.logger.info("📅 Not a market day, skipping fundamental download")
            return False
        
        # Check if we need to run (e.g., weekly on Monday)
        if datetime.now().weekday() == 0:  # Monday
            return True
        
        # Check if it's been more than 7 days since last run
        progress = self.collector.progress
        if 'last_updated' in progress:
            last_updated = datetime.fromisoformat(progress['last_updated'])
            days_since_update = (datetime.now() - last_updated).days
            if days_since_update >= 7:
                return True
        
        return False
    
    def run_fundamental_download(self):
        """Run fundamental data download."""
        if not self._should_run_download():
            self.logger.info("⏰ Skipping fundamental download - not required")
            return
        
        try:
            self.logger.info("🚀 Starting scheduled fundamental data download")
            
            # Get current status
            status = self.collector.get_download_status()
            self.logger.info(f"📊 Current status: {status['completed_symbols']}/{status['total_symbols']} completed")
            
            # Run download with smaller batch size for scheduled runs
            self.collector.download_all_fundamentals(batch_size=30)
            
            # Log completion
            final_status = self.collector.get_download_status()
            self.logger.info(f"✅ Scheduled download completed: {final_status['completed_symbols']} success, {final_status['failed_symbols']} failed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in scheduled fundamental download: {e}")
    
    def run_incremental_update(self):
        """Run incremental update for recent filings."""
        try:
            self.logger.info("🔄 Starting incremental fundamental update")
            
            # This could be enhanced to only download recent filings
            # For now, we'll run a smaller batch
            self.collector.download_all_fundamentals(batch_size=20)
            
            self.logger.info("✅ Incremental update completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in incremental update: {e}")
    
    def setup_schedule(self):
        """Setup the scheduling for fundamental data collection."""
        self.logger.info("📅 Setting up fundamental data collection schedule...")
        
        # Weekly full download on Monday at 6 AM
        schedule.every().monday.at("06:00").do(self.run_fundamental_download)
        
        # Daily incremental update at 8 PM (for any new filings)
        schedule.every().day.at("20:00").do(self.run_incremental_update)
        
        # Health check every day at 10 AM
        schedule.every().day.at("10:00").do(self.check_health)
        
        self.logger.info("✅ Schedule setup completed")
        self.logger.info("   Full download: Monday 6:00 AM")
        self.logger.info("   Incremental update: Daily 8:00 PM")
        self.logger.info("   Health check: Daily 10:00 AM")
    
    def check_health(self):
        """Check the health of fundamental data."""
        try:
            status = self.collector.get_download_status()
            
            self.logger.info(f"📊 Fundamental Data Health Check:")
            self.logger.info(f"   Total Symbols: {status['total_symbols']}")
            self.logger.info(f"   Completed: {status['completed_symbols']}")
            self.logger.info(f"   Failed: {status['failed_symbols']}")
            self.logger.info(f"   Progress: {status['progress_percentage']:.1f}%")
            
            # Alert if too many failures
            if status['failed_symbols'] > status['total_symbols'] * 0.1:  # More than 10% failed
                self.logger.warning("⚠️ High failure rate detected in fundamental data collection")
            
            # Alert if no recent updates
            if status['progress_percentage'] < 50:
                self.logger.warning("⚠️ Low completion rate in fundamental data collection")
                
        except Exception as e:
            self.logger.error(f"❌ Error checking health: {e}")
    
    def run_scheduler(self):
        """Run the scheduler loop."""
        self.logger.info("🚀 Starting fundamental data scheduler...")
        
        # Setup schedule
        self.setup_schedule()
        
        # If it's Monday and before 6 AM, run initial download
        if self._is_market_day() and datetime.now().weekday() == 0 and datetime.now().hour < 6:
            self.logger.info("⏰ Monday morning detected, running initial download...")
            self.run_fundamental_download()
        
        # Run scheduler loop
        self.running = True
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Scheduler stopped by user")
                self.running = False
                break
                
            except Exception as e:
                self.logger.error(f"❌ Error in scheduler loop: {e}")
                time.sleep(60)
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.logger.info("🛑 Stopping fundamental data scheduler...")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the scheduler and fundamental data."""
        status = {
            'scheduler_running': self.running,
            'market_day': self._is_market_day(),
            'should_run_download': self._should_run_download(),
            'fundamental_status': self.collector.get_download_status()
        }
        
        return status


def main():
    """Main function for running the scheduler."""
    scheduler = FundamentalScheduler()
    
    try:
        scheduler.run_scheduler()
    except Exception as e:
        print(f"❌ Error running scheduler: {e}")


if __name__ == "__main__":
    main()
