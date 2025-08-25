#!/usr/bin/env python3
"""
Options Data Collection Scheduler
Service script to manage options chain data collection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import signal
import argparse
from datetime import datetime
from loguru import logger
import psutil

from options_chain_collector import OptionsChainCollector


class OptionsScheduler:
    """Service scheduler for options data collection."""
    
    def __init__(self, db_path: str = "data/options_chain_data.duckdb"):
        """
        Initialize the options scheduler.
        
        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = db_path
        self.collector = None
        self.running = False
        self.pid_file = "options_collector.pid"
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Options Scheduler initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"📡 Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def _write_pid_file(self):
        """Write PID to file for service management."""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.info(f"📝 PID written to {self.pid_file}")
        except Exception as e:
            logger.error(f"❌ Error writing PID file: {e}")
    
    def _remove_pid_file(self):
        """Remove PID file."""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
                logger.info(f"🗑️ Removed PID file {self.pid_file}")
        except Exception as e:
            logger.error(f"❌ Error removing PID file: {e}")
    
    def _check_if_running(self) -> bool:
        """Check if another instance is already running."""
        try:
            if os.path.exists(self.pid_file):
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is actually running
                if psutil.pid_exists(pid):
                    logger.warning(f"⚠️ Options collector already running with PID {pid}")
                    return True
                else:
                    # Remove stale PID file
                    self._remove_pid_file()
                    return False
            return False
        except Exception as e:
            logger.error(f"❌ Error checking if running: {e}")
            return False
    
    def start(self):
        """Start the options data collection service."""
        try:
            if self._check_if_running():
                logger.error("❌ Another instance is already running")
                return False
            
            logger.info("🚀 Starting Options Data Collection Service")
            
            # Write PID file
            self._write_pid_file()
            
            # Initialize collector
            self.collector = OptionsChainCollector(self.db_path)
            self.running = True
            
            # Start the scheduler
            self.collector.start_scheduler()
            
        except Exception as e:
            logger.error(f"❌ Error starting service: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the options data collection service."""
        try:
            logger.info("🛑 Stopping Options Data Collection Service")
            self.running = False
            
            # Remove PID file
            self._remove_pid_file()
            
            logger.info("✅ Service stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping service: {e}")
    
    def status(self) -> bool:
        """Check service status."""
        try:
            if self._check_if_running():
                logger.info("✅ Options collector service is running")
                return True
            else:
                logger.info("❌ Options collector service is not running")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking status: {e}")
            return False
    
    def restart(self):
        """Restart the service."""
        try:
            logger.info("🔄 Restarting Options Data Collection Service")
            self.stop()
            time.sleep(2)  # Wait for cleanup
            self.start()
        except Exception as e:
            logger.error(f"❌ Error restarting service: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Options Data Collection Scheduler')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'], 
                       help='Action to perform')
    parser.add_argument('--db-path', default='data/options_chain_data.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon process')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/options_collector.log", rotation="1 day", retention="7 days", level="DEBUG")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize scheduler
    scheduler = OptionsScheduler(args.db_path)
    
    try:
        if args.action == 'start':
            if args.daemon:
                # Fork to background
                pid = os.fork()
                if pid > 0:
                    # Parent process
                    logger.info(f"🚀 Started daemon process with PID {pid}")
                    return
                else:
                    # Child process
                    os.setsid()
                    os.umask(0)
                    
                    # Redirect file descriptors
                    sys.stdout.flush()
                    sys.stderr.flush()
                    
                    with open('/dev/null', 'r') as f:
                        os.dup2(f.fileno(), sys.stdin.fileno())
                    with open('/dev/null', 'a+') as f:
                        os.dup2(f.fileno(), sys.stdout.fileno())
                    with open('/dev/null', 'a+') as f:
                        os.dup2(f.fileno(), sys.stderr.fileno())
            
            scheduler.start()
            
        elif args.action == 'stop':
            scheduler.stop()
            
        elif args.action == 'restart':
            scheduler.restart()
            
        elif args.action == 'status':
            scheduler.status()
            
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        scheduler.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        scheduler.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
