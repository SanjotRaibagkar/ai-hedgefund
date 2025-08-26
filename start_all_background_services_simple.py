#!/usr/bin/env python3
"""
Simplified Unified Background Services Manager
Manages all 3 background services without encoding issues.
"""

import os
import sys
import time
import signal
import subprocess
import psutil
from datetime import datetime
import threading

class SimpleBackgroundServiceManager:
    """Simplified manager for all background services."""
    
    def __init__(self):
        """Initialize the service manager."""
        self.services = {
            'options_analysis': {
                'script': 'start_options_analysis_background.py',
                'name': 'Options Analysis Scheduler',
                'process': None,
                'pid': None
            },
            'intraday_collection': {
                'script': 'start_intraday_data_collector_fixed.py',
                'name': 'Intraday Data Collection',
                'process': None,
                'pid': None
            },
            'options_collection': {
                'script': 'start_options_collector_background.py',
                'name': 'Options Chain Collection',
                'process': None,
                'pid': None
            }
        }
        
        self.running = True
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"Received signal {signum}, shutting down...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)
    
    def find_existing_processes(self):
        """Find existing processes by script name."""
        existing_pids = {}
        
        for service_key, service_info in self.services.items():
            script_name = service_info['script']
            found_pid = None
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any(script_name in cmd for cmd in proc.info['cmdline']):
                        found_pid = proc.info['pid']
                        print(f"Found existing {service_info['name']} process: PID {found_pid}")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            existing_pids[service_key] = found_pid
        
        return existing_pids
    
    def is_process_running(self, pid):
        """Check if a process is running."""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def start_service(self, service_key):
        """Start a specific service."""
        service_info = self.services[service_key]
        
        try:
            print(f"Starting {service_info['name']}...")
            
            # Set environment variables for proper encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
            
            # Start the process
            process = subprocess.Popen(
                ['poetry', 'run', 'python', service_info['script']],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
                env=env
            )
            
            # Wait a moment for process to start
            time.sleep(3)
            
            # Check if process started successfully
            if process.poll() is None:  # Process is still running
                service_info['process'] = process
                service_info['pid'] = process.pid
                print(f"SUCCESS: {service_info['name']} started with PID: {process.pid}")
                return True
            else:
                print(f"FAILED: {service_info['name']} failed to start")
                return False
                
        except Exception as e:
            print(f"ERROR starting {service_info['name']}: {e}")
            return False
    
    def check_service_health(self, service_key):
        """Check if a service is healthy and running."""
        service_info = self.services[service_key]
        
        if not service_info['process']:
            return False
        
        try:
            # Check if process is still running
            if service_info['process'].poll() is not None:
                print(f"WARNING: {service_info['name']} process has stopped")
                service_info['process'] = None
                service_info['pid'] = None
                return False
            
            # Check if process is responsive
            if service_info['pid']:
                if not self.is_process_running(service_info['pid']):
                    print(f"WARNING: {service_info['name']} PID {service_info['pid']} not found")
                    service_info['process'] = None
                    service_info['pid'] = None
                    return False
            
            return True
            
        except Exception as e:
            print(f"ERROR checking {service_info['name']} health: {e}")
            return False
    
    def monitor_and_restart_services(self):
        """Monitor all services and restart if needed."""
        while self.running:
            try:
                for service_key, service_info in self.services.items():
                    if not self.check_service_health(service_key):
                        print(f"RESTARTING: {service_info['name']}...")
                        self.start_service(service_key)
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"ERROR in service monitoring: {e}")
                time.sleep(60)
    
    def start_all_services(self):
        """Start all services intelligently."""
        print("Starting Unified Background Services Manager")
        print("=" * 60)
        
        # Find existing processes
        existing_pids = self.find_existing_processes()
        
        # Start services that aren't already running
        for service_key, service_info in self.services.items():
            existing_pid = existing_pids.get(service_key)
            
            if existing_pid and self.is_process_running(existing_pid):
                print(f"SUCCESS: {service_info['name']} already running (PID: {existing_pid})")
                service_info['pid'] = existing_pid
            else:
                print(f"STARTING: {service_info['name']}...")
                if not self.start_service(service_key):
                    print(f"FAILED: Could not start {service_info['name']}")
        
        print("=" * 60)
        print("All services initialized")
    
    def stop_all_services(self):
        """Stop all services gracefully."""
        print("Stopping all background services...")
        
        for service_key, service_info in self.services.items():
            if service_info['process']:
                try:
                    print(f"Stopping {service_info['name']} (PID: {service_info['pid']})...")
                    service_info['process'].terminate()
                    
                    try:
                        service_info['process'].wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        service_info['process'].kill()
                    
                    service_info['process'] = None
                    service_info['pid'] = None
                    print(f"SUCCESS: {service_info['name']} stopped")
                    
                except Exception as e:
                    print(f"ERROR stopping {service_info['name']}: {e}")
        
        print("All services stopped")
    
    def print_status(self):
        """Print current status of all services."""
        print("\n" + "=" * 80)
        print("BACKGROUND SERVICES STATUS")
        print("=" * 80)
        
        for service_key, service_info in self.services.items():
            is_running = self.check_service_health(service_key)
            status = "RUNNING" if is_running else "STOPPED"
            pid_info = f"PID: {service_info['pid']}" if service_info['pid'] else "Not Running"
            
            print(f"{status:<10} {service_info['name']:<25} | {pid_info}")
        
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
            print("Received keyboard interrupt")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.stop_all_services()


def main():
    """Main function."""
    print("Unified Background Services Manager (Simplified)")
    print("=" * 50)
    print("Managing: Options Analysis, Intraday Collection, Options Collection")
    print("=" * 50)
    
    manager = SimpleBackgroundServiceManager()
    
    try:
        manager.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
