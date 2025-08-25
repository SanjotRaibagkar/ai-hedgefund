#!/usr/bin/env python3
"""
Auto Options Data Collector
Enhanced options collector with automatic startup and monitoring
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import schedule
import psutil
import signal
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from loguru import logger
import json

from .options_chain_collector import OptionsChainCollector


class AutoOptionsCollector:
    """Enhanced options collector with automatic startup and monitoring."""
    
    def __init__(self, db_path: str = "data/options_chain_data.duckdb"):
        """Initialize the auto options collector."""
        self.db_path = db_path
        self.collector = OptionsChainCollector(db_path)
        self.running = False
        self.monitoring_thread = None
        self.status_file = "data/options_collector_status.json"
        
        # Trading hours (IST)
        self.market_open = "09:30"
        self.market_close = "15:30"
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Auto Options Collector initialized")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        
    def _save_status(self, status: Dict):
        """Save collector status to file."""
        try:
            os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Error saving status: {e}")
            
    def _load_status(self) -> Dict:
        """Load collector status from file."""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading status: {e}")
        return {}
        
    def _update_status(self, **kwargs):
        """Update collector status."""
        status = self._load_status()
        status.update({
            'last_updated': datetime.now().isoformat(),
            'running': self.running,
            'market_hours': self.collector._is_market_hours(),
            'trading_day': self.collector._is_trading_day(),
            **kwargs
        })
        self._save_status(status)
        
    def _monitoring_loop(self):
        """Monitoring loop to track collector health."""
        while self.running:
            try:
                # Check if collector is still running
                if not self.collector.connection:
                    logger.error("❌ Database connection lost, attempting to reconnect...")
                    self.collector = OptionsChainCollector(self.db_path)
                
                # Update status
                self._update_status(
                    memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                    cpu_percent=psutil.Process().cpu_percent()
                )
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(60)
                
    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("📊 Monitoring started")
            
    def schedule_startup(self):
        """Schedule automatic startup at market open."""
        schedule.every().day.at("09:30").do(self.start_collection)
        schedule.every().day.at("15:30").do(self.stop_collection)
        
        logger.info("⏰ Scheduled startup at 9:30 AM daily")
        logger.info("⏰ Scheduled shutdown at 3:30 PM daily")
        
    def start_collection(self):
        """Start the options data collection."""
        if not self.running:
            logger.info("🚀 Starting options data collection...")
            self.running = True
            self.start_monitoring()
            
            # Start the collector
            self.collector.start_scheduler()
            
    def stop_collection(self):
        """Stop the options data collection."""
        if self.running:
            logger.info("🛑 Stopping options data collection...")
            self.running = False
            self._update_status(running=False)
            
    def stop(self):
        """Stop the auto collector."""
        self.stop_collection()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            
    def run_with_schedule(self):
        """Run the collector with automatic scheduling."""
        logger.info("🚀 Starting Auto Options Collector with scheduling")
        logger.info(f"📅 Trading hours: {self.market_open} - {self.market_close} IST")
        
        # Schedule startup and shutdown
        self.schedule_startup()
        
        # If it's currently market hours, start immediately
        if self.collector._is_market_hours() and self.collector._is_trading_day():
            logger.info("⏰ Currently market hours, starting collection immediately")
            self.start_collection()
        
        # Run the scheduler
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(60)
                
        self.stop()
        
    def run_continuous(self):
        """Run the collector continuously (for manual control)."""
        logger.info("🚀 Starting Auto Options Collector in continuous mode")
        
        # Start monitoring
        self.start_monitoring()
        
        # Start collection if it's market hours
        if self.collector._is_market_hours() and self.collector._is_trading_day():
            self.start_collection()
        else:
            logger.info("⏰ Outside market hours, waiting for market open...")
            
        # Keep running
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
            self.stop()


def main():
    """Main function to run the auto options collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Options Data Collector")
    parser.add_argument("--mode", choices=["schedule", "continuous"], default="schedule",
                       help="Run mode: schedule (auto start/stop) or continuous (manual control)")
    parser.add_argument("--start-now", action="store_true",
                       help="Start collection immediately regardless of market hours")
    
    args = parser.parse_args()
    
    try:
        auto_collector = AutoOptionsCollector()
        
        if args.start_now:
            logger.info("🎯 Starting collection immediately (--start-now flag)")
            auto_collector.start_collection()
            auto_collector.run_continuous()
        elif args.mode == "schedule":
            auto_collector.run_with_schedule()
        else:
            auto_collector.run_continuous()
            
    except Exception as e:
        logger.error(f"❌ Fatal error in Auto Options Collector: {e}")
        raise


if __name__ == "__main__":
    main()
