#!/usr/bin/env python3
"""
Unified Background Services Manager
Manages all 3 background services with intelligent process management:
1. Options Analysis Scheduler
2. Intraday Data Collection
3. Options Chain Collection

Features:
- Prevents duplicate processes
- Auto-restart if processes stop
- Comprehensive logging
- Process health monitoring
"""

import os
import sys
import time
import signal
import subprocess
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Optional
import threading

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from loguru import logger

class BackgroundServiceManager:
    """Unified manager for all background services."""
    
    def __init__(self):
        """Initialize the service manager."""
        self.logger = logger
        self.services = {
            'options_analysis': {
                'script': 'start_options_analysis_background.py',
                'name': 'Options Analysis Scheduler',
                'log_file': 'logs/options_analysis_background.log',
                'process': None,
                'pid': None,
                'last_check': None
            },
            'intraday_collection': {
                'script': 'start_intraday_data_collector_fixed.py',
                'name': 'Intraday Data Collection',
                'log_file': 'logs/intraday_data_collection_background.log',
                'process': None,
                'pid': None,
                'last_check': None
            },
            'options_collection': {
                'script': 'start_options_collector_background.py',
                'name': 'Options Chain Collection',
                'log_file': 'logs/options_collector_background.log',
                'process': None,
                'pid': None,
                'last_check': None
            }
        }
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Flag to control the main loop
        self.running = True
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # Add file handler
        logger.add(
            "logs/unified_background_manager.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="1 day",
            retention="7 days"
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)
    
    def find_existing_processes(self) -> Dict[str, Optional[int]]:
        """Find existing processes by script name."""
        existing_pids = {}
        
        for service_key, service_info in self.services.items():
            script_name = service_info['script']
            found_pid = None
            
            # Search for processes running the script
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any(script_name in cmd for cmd in proc.info['cmdline']):
                        found_pid = proc.info['pid']
                        self.logger.info(f"🔍 Found existing {service_info['name']} process: PID {found_pid}")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            existing_pids[service_key] = found_pid
        
        return existing_pids
    
    def is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def start_service(self, service_key: str) -> bool:
        """Start a specific service."""
        service_info = self.services[service_key]
        
        try:
            self.logger.info(f"🚀 Starting {service_info['name']}...")
            
            # Start the process with proper encoding for Windows
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
            
            process = subprocess.Popen(
                ['poetry', 'run', 'python', service_info['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
                env=env
            )
            
            # Wait a moment for process to start
            time.sleep(2)
            
            # Check if process started successfully
            if process.poll() is None:  # Process is still running
                service_info['process'] = process
                service_info['pid'] = process.pid
                service_info['last_check'] = datetime.now()
                
                self.logger.info(f"✅ {service_info['name']} started successfully with PID: {process.pid}")
                return True
            else:
                # Get error output if process failed
                stdout, stderr = process.communicate()
                if stderr:
                    self.logger.error(f"❌ {service_info['name']} failed to start. Error: {stderr.decode()}")
                else:
                    self.logger.error(f"❌ Failed to start {service_info['name']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting {service_info['name']}: {e}")
            return False
    
    def stop_service(self, service_key: str) -> bool:
        """Stop a specific service."""
        service_info = self.services[service_key]
        
        if service_info['process']:
            try:
                self.logger.info(f"🛑 Stopping {service_info['name']} (PID: {service_info['pid']})...")
                service_info['process'].terminate()
                
                # Wait for graceful shutdown
                try:
                    service_info['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"⚠️ Force killing {service_info['name']}...")
                    service_info['process'].kill()
                
                service_info['process'] = None
                service_info['pid'] = None
                service_info['last_check'] = None
                
                self.logger.info(f"✅ {service_info['name']} stopped successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Error stopping {service_info['name']}: {e}")
                return False
        
        return True
    
    def check_service_health(self, service_key: str) -> bool:
        """Check if a service is healthy and running."""
        service_info = self.services[service_key]
        
        if not service_info['process']:
            return False
        
        try:
            # Check if process is still running
            if service_info['process'].poll() is not None:
                # Get error output if process failed
                stdout, stderr = service_info['process'].communicate()
                if stderr:
                    self.logger.warning(f"⚠️ {service_info['name']} process has stopped. Error: {stderr.decode()}")
                else:
                    self.logger.warning(f"⚠️ {service_info['name']} process has stopped")
                service_info['process'] = None
                service_info['pid'] = None
                return False
            
            # Check if process is responsive
            if service_info['pid']:
                if not self.is_process_running(service_info['pid']):
                    self.logger.warning(f"⚠️ {service_info['name']} PID {service_info['pid']} not found")
                    service_info['process'] = None
                    service_info['pid'] = None
                    return False
            
            # Update last check time
            service_info['last_check'] = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking {service_info['name']} health: {e}")
            return False
    
    def monitor_and_restart_services(self):
        """Monitor all services and restart if needed."""
        while self.running:
            try:
                for service_key, service_info in self.services.items():
                    if not self.check_service_health(service_key):
                        self.logger.info(f"🔄 Restarting {service_info['name']}...")
                        self.start_service(service_key)
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in service monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start_all_services(self):
        """Start all services intelligently."""
        self.logger.info("🚀 Starting Unified Background Services Manager")
        self.logger.info("=" * 60)
        
        # Find existing processes
        existing_pids = self.find_existing_processes()
        
        # Start services that aren't already running
        for service_key, service_info in self.services.items():
            existing_pid = existing_pids.get(service_key)
            
            if existing_pid and self.is_process_running(existing_pid):
                self.logger.info(f"✅ {service_info['name']} already running (PID: {existing_pid})")
                service_info['pid'] = existing_pid
                service_info['last_check'] = datetime.now()
            else:
                self.logger.info(f"🔄 Starting {service_info['name']}...")
                if not self.start_service(service_key):
                    self.logger.error(f"❌ Failed to start {service_info['name']}")
        
        self.logger.info("=" * 60)
        self.logger.info("✅ All services initialized")
    
    def stop_all_services(self):
        """Stop all services gracefully."""
        self.logger.info("🛑 Stopping all background services...")
        
        for service_key in self.services:
            self.stop_service(service_key)
        
        self.logger.info("✅ All services stopped")
    
    def get_status_report(self) -> Dict[str, Dict]:
        """Get comprehensive status report of all services."""
        status = {}
        
        for service_key, service_info in self.services.items():
            is_running = self.check_service_health(service_key)
            
            status[service_key] = {
                'name': service_info['name'],
                'running': is_running,
                'pid': service_info['pid'],
                'last_check': service_info['last_check'],
                'uptime': None
            }
            
            # Calculate uptime if running
            if is_running and service_info['last_check']:
                uptime = datetime.now() - service_info['last_check']
                status[service_key]['uptime'] = str(uptime).split('.')[0]
        
        return status
    
    def print_status(self):
        """Print current status of all services."""
        status = self.get_status_report()
        
        print("\n" + "=" * 80)
        print("📊 UNIFIED BACKGROUND SERVICES STATUS")
        print("=" * 80)
        
        for service_key, info in status.items():
            status_icon = "✅" if info['running'] else "❌"
            pid_info = f"PID: {info['pid']}" if info['pid'] else "Not Running"
            uptime_info = f"Uptime: {info['uptime']}" if info['uptime'] else ""
            
            print(f"{status_icon} {info['name']:<25} | {pid_info:<15} | {uptime_info}")
        
        print("=" * 80)
    
    def run(self):
        """Main run loop."""
        try:
            # Start all services
            self.start_all_services()
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self.monitor_and_restart_services, daemon=True)
            monitor_thread.start()
            
            # Main loop - print status every 5 minutes
            last_status_print = datetime.now()
            
            while self.running:
                time.sleep(60)  # Check every minute
                
                # Print status every 5 minutes
                if (datetime.now() - last_status_print).seconds >= 300:
                    self.print_status()
                    last_status_print = datetime.now()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.stop_all_services()


def main():
    """Main function."""
    print("🚀 Unified Background Services Manager")
    print("=" * 50)
    print("Managing: Options Analysis, Intraday Collection, Options Collection")
    print("=" * 50)
    
    manager = BackgroundServiceManager()
    
    try:
        manager.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
