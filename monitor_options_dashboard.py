#!/usr/bin/env python3
"""
Options Data Collection Monitoring Dashboard
Comprehensive monitoring and status dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from src.data.downloaders.options_chain_collector import OptionsChainCollector


class OptionsMonitoringDashboard:
    """Comprehensive monitoring dashboard for options data collection."""
    
    def __init__(self):
        """Initialize the monitoring dashboard."""
        self.collector = OptionsChainCollector()
        self.status_file = "data/options_collector_status.json"
        
    def get_collector_status(self) -> Dict:
        """Get the current collector status."""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error reading status file: {e}")
        return {}
        
    def get_process_info(self) -> Dict:
        """Get process information."""
        try:
            # Find the options collector process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'start_options_collector.py' in cmdline or 'auto_options_collector.py' in cmdline:
                        return {
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'cpu_percent': proc.info['cpu_percent'],
                            'status': proc.status(),
                            'create_time': datetime.fromtimestamp(proc.create_time()).isoformat()
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"❌ Error getting process info: {e}")
        return {}
        
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            # Get recent data counts
            nifty_recent = self.collector.get_recent_data('NIFTY', minutes=60)
            banknifty_recent = self.collector.get_recent_data('BANKNIFTY', minutes=60)
            
            # Get daily summaries
            nifty_summary = self.collector.get_daily_summary('NIFTY')
            banknifty_summary = self.collector.get_daily_summary('BANKNIFTY')
            
            return {
                'nifty_recent_records': len(nifty_recent),
                'banknifty_recent_records': len(banknifty_recent),
                'nifty_daily_total': nifty_summary.get('total_records', 0),
                'banknifty_daily_total': banknifty_summary.get('total_records', 0),
                'nifty_avg_spot': nifty_summary.get('avg_spot_price', 0),
                'banknifty_avg_spot': banknifty_summary.get('avg_spot_price', 0)
            }
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}
            
    def get_market_status(self) -> Dict:
        """Get current market status."""
        try:
            is_trading_day = self.collector._is_trading_day()
            is_market_hours = self.collector._is_market_hours()
            
            now = datetime.now()
            market_open = datetime.strptime("09:30", "%H:%M").time()
            market_close = datetime.strptime("15:30", "%H:%M").time()
            
            # Calculate time until market open/close
            if now.time() < market_open:
                time_until_open = datetime.combine(now.date(), market_open) - now
                next_event = f"Market opens in {time_until_open}"
            elif now.time() < market_close:
                time_until_close = datetime.combine(now.date(), market_close) - now
                next_event = f"Market closes in {time_until_close}"
            else:
                # Calculate time until next trading day
                next_trading_day = now + timedelta(days=1)
                while next_trading_day.weekday() >= 5:  # Weekend
                    next_trading_day += timedelta(days=1)
                time_until_next = next_trading_day - now
                next_event = f"Next trading day in {time_until_next}"
                
            return {
                'is_trading_day': is_trading_day,
                'is_market_hours': is_market_hours,
                'current_time': now.isoformat(),
                'next_event': next_event
            }
        except Exception as e:
            logger.error(f"❌ Error getting market status: {e}")
            return {}
            
    def get_system_info(self) -> Dict:
        """Get system information."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('.').percent,
                'uptime': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting system info: {e}")
            return {}
            
    def check_service_status(self) -> Dict:
        """Check Windows service status."""
        try:
            result = subprocess.run(['sc', 'query', 'AIHedgeFundOptionsCollector'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                status = {}
                for line in lines:
                    if 'STATE' in line:
                        status['service_state'] = line.split(':')[1].strip()
                    elif 'START_TYPE' in line:
                        status['start_type'] = line.split(':')[1].strip()
                return status
        except Exception as e:
            logger.error(f"❌ Error checking service status: {e}")
        return {}
        
    def check_task_scheduler(self) -> Dict:
        """Check Windows Task Scheduler status."""
        try:
            result = subprocess.run(['schtasks', '/query', '/tn', 'AIHedgeFundOptionsCollector'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                status = {}
                for line in lines:
                    if 'Status' in line:
                        status['task_status'] = line.split(':')[1].strip()
                    elif 'Next Run Time' in line:
                        status['next_run'] = line.split(':')[1].strip()
                return status
        except Exception as e:
            logger.error(f"❌ Error checking task scheduler: {e}")
        return {}
        
    def display_dashboard(self):
        """Display the monitoring dashboard."""
        print("\n" + "="*80)
        print("🎯 OPTIONS DATA COLLECTION MONITORING DASHBOARD")
        print("="*80)
        
        # Market Status
        market_status = self.get_market_status()
        print(f"\n📅 MARKET STATUS:")
        print(f"   Trading Day: {'✅ Yes' if market_status.get('is_trading_day') else '❌ No'}")
        print(f"   Market Hours: {'✅ Yes' if market_status.get('is_market_hours') else '❌ No'}")
        print(f"   Current Time: {market_status.get('current_time', 'Unknown')}")
        print(f"   Next Event: {market_status.get('next_event', 'Unknown')}")
        
        # Collector Status
        collector_status = self.get_collector_status()
        print(f"\n🚀 COLLECTOR STATUS:")
        print(f"   Running: {'✅ Yes' if collector_status.get('running') else '❌ No'}")
        print(f"   Last Updated: {collector_status.get('last_updated', 'Unknown')}")
        if 'memory_usage' in collector_status:
            print(f"   Memory Usage: {collector_status['memory_usage']:.1f} MB")
        if 'cpu_percent' in collector_status:
            print(f"   CPU Usage: {collector_status['cpu_percent']:.1f}%")
            
        # Process Information
        process_info = self.get_process_info()
        if process_info:
            print(f"\n⚙️ PROCESS INFORMATION:")
            print(f"   PID: {process_info.get('pid', 'Unknown')}")
            print(f"   Status: {process_info.get('status', 'Unknown')}")
            print(f"   Memory: {process_info.get('memory_mb', 0):.1f} MB")
            print(f"   CPU: {process_info.get('cpu_percent', 0):.1f}%")
            print(f"   Started: {process_info.get('create_time', 'Unknown')}")
        else:
            print(f"\n⚙️ PROCESS INFORMATION:")
            print(f"   ❌ No options collector process found")
            
        # Database Statistics
        db_stats = self.get_database_stats()
        print(f"\n📊 DATABASE STATISTICS:")
        print(f"   NIFTY Recent (1h): {db_stats.get('nifty_recent_records', 0):,} records")
        print(f"   BANKNIFTY Recent (1h): {db_stats.get('banknifty_recent_records', 0):,} records")
        print(f"   NIFTY Daily Total: {db_stats.get('nifty_daily_total', 0):,} records")
        print(f"   BANKNIFTY Daily Total: {db_stats.get('banknifty_daily_total', 0):,} records")
        print(f"   NIFTY Avg Spot: ₹{db_stats.get('nifty_avg_spot', 0):,.2f}")
        print(f"   BANKNIFTY Avg Spot: ₹{db_stats.get('banknifty_avg_spot', 0):,.2f}")
        
        # System Information
        system_info = self.get_system_info()
        print(f"\n💻 SYSTEM INFORMATION:")
        print(f"   CPU Usage: {system_info.get('cpu_percent', 0):.1f}%")
        print(f"   Memory Usage: {system_info.get('memory_percent', 0):.1f}%")
        print(f"   Disk Usage: {system_info.get('disk_usage', 0):.1f}%")
        print(f"   System Uptime: {system_info.get('uptime', 'Unknown')}")
        
        # Service Status
        service_status = self.check_service_status()
        if service_status:
            print(f"\n🔧 WINDOWS SERVICE:")
            print(f"   State: {service_status.get('service_state', 'Unknown')}")
            print(f"   Start Type: {service_status.get('start_type', 'Unknown')}")
        else:
            print(f"\n🔧 WINDOWS SERVICE:")
            print(f"   ❌ Service not found or not accessible")
            
        # Task Scheduler Status
        task_status = self.check_task_scheduler()
        if task_status:
            print(f"\n⏰ TASK SCHEDULER:")
            print(f"   Status: {task_status.get('task_status', 'Unknown')}")
            print(f"   Next Run: {task_status.get('next_run', 'Unknown')}")
        else:
            print(f"\n⏰ TASK SCHEDULER:")
            print(f"   ❌ Task not found or not accessible")
            
        print("\n" + "="*80)
        
    def run_continuous_monitoring(self, interval: int = 60):
        """Run continuous monitoring with specified interval."""
        logger.info(f"📊 Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                self.display_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")


def main():
    """Main function for the monitoring dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Options Data Collection Monitoring Dashboard")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=60,
                       help="Monitoring interval in seconds (default: 60)")
    
    args = parser.parse_args()
    
    try:
        dashboard = OptionsMonitoringDashboard()
        
        if args.continuous:
            dashboard.run_continuous_monitoring(args.interval)
        else:
            dashboard.display_dashboard()
            
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        raise


if __name__ == "__main__":
    main()
