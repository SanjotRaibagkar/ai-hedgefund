#!/usr/bin/env python3
"""
Simple Options Data Collection Status Checker
Quick status check without database access
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import psutil
import subprocess
from datetime import datetime, timedelta
from loguru import logger


def check_process_status():
    """Check if options collector process is running."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'start_options_collector.py' in cmdline or 'auto_options_collector.py' in cmdline:
                    return {
                        'running': True,
                        'pid': proc.info['pid'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'cpu_percent': proc.info['cpu_percent'],
                        'status': proc.status(),
                        'create_time': datetime.fromtimestamp(proc.create_time()).isoformat()
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"❌ Error checking process: {e}")
    return {'running': False}


def check_market_status():
    """Check current market status."""
    try:
        now = datetime.now()
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        
        # Check if it's weekend
        is_weekend = now.weekday() >= 5
        
        # Check if it's market hours
        is_market_hours = market_open <= now.time() <= market_close
        
        # Calculate next event
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
            'is_weekend': is_weekend,
            'is_market_hours': is_market_hours,
            'current_time': now.isoformat(),
            'next_event': next_event
        }
    except Exception as e:
        logger.error(f"❌ Error checking market status: {e}")
        return {}


def check_status_file():
    """Check status file if it exists."""
    status_file = "data/options_collector_status.json"
    try:
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error reading status file: {e}")
    return {}


def display_status():
    """Display the current status."""
    print("\n" + "="*80)
    print("🎯 OPTIONS DATA COLLECTION STATUS")
    print("="*80)
    
    # Process Status
    process_info = check_process_status()
    print(f"\n🚀 PROCESS STATUS:")
    if process_info.get('running'):
        print(f"   ✅ Running (PID: {process_info.get('pid')})")
        print(f"   Memory: {process_info.get('memory_mb', 0):.1f} MB")
        print(f"   CPU: {process_info.get('cpu_percent', 0):.1f}%")
        print(f"   Started: {process_info.get('create_time', 'Unknown')}")
    else:
        print(f"   ❌ Not running")
    
    # Market Status
    market_status = check_market_status()
    print(f"\n📅 MARKET STATUS:")
    print(f"   Weekend: {'✅ Yes' if market_status.get('is_weekend') else '❌ No'}")
    print(f"   Market Hours: {'✅ Yes' if market_status.get('is_market_hours') else '❌ No'}")
    print(f"   Current Time: {market_status.get('current_time', 'Unknown')}")
    print(f"   Next Event: {market_status.get('next_event', 'Unknown')}")
    
    # Status File
    status_data = check_status_file()
    if status_data:
        print(f"\n📊 COLLECTOR STATUS:")
        print(f"   Running: {'✅ Yes' if status_data.get('running') else '❌ No'}")
        print(f"   Last Updated: {status_data.get('last_updated', 'Unknown')}")
        if 'memory_usage' in status_data:
            print(f"   Memory Usage: {status_data['memory_usage']:.1f} MB")
        if 'cpu_percent' in status_data:
            print(f"   CPU Usage: {status_data['cpu_percent']:.1f}%")
    
    # System Info
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        print(f"\n💻 SYSTEM INFO:")
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   Memory Usage: {memory_percent:.1f}%")
    except Exception as e:
        logger.error(f"❌ Error getting system info: {e}")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Options Data Collection Status Checker")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Monitoring interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    try:
        if args.continuous:
            logger.info(f"📊 Starting continuous monitoring (interval: {args.interval}s)")
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                display_status()
                time.sleep(args.interval)
        else:
            display_status()
            
    except KeyboardInterrupt:
        logger.info("🛑 Status checking stopped by user")
    except Exception as e:
        logger.error(f"❌ Status checker error: {e}")
        raise


if __name__ == "__main__":
    main()
