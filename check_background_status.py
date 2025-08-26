#!/usr/bin/env python3
"""
Background Services Status Checker
Quick script to check the status of all background services.
"""

import os
import sys
import psutil
from datetime import datetime

def find_service_processes():
    """Find all background service processes."""
    services = {
        'options_analysis': 'start_options_analysis_background.py',
        'intraday_collection': 'start_intraday_data_collector_fixed.py',
        'options_collection': 'start_options_collector_background.py'
    }
    
    found_processes = {}
    
    for service_name, script_name in services.items():
        found_pid = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any(script_name in cmd for cmd in proc.info['cmdline']):
                    found_pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        found_processes[service_name] = found_pid
    
    return found_processes

def check_process_health(pid):
    """Check if a process is healthy."""
    if not pid:
        return False, None
    
    try:
        process = psutil.Process(pid)
        if process.is_running():
            create_time = datetime.fromtimestamp(process.create_time())
            uptime = datetime.now() - create_time
            return True, uptime
        else:
            return False, None
    except psutil.NoSuchProcess:
        return False, None

def main():
    """Main function."""
    print("📊 Background Services Status Check")
    print("=" * 60)
    
    processes = find_service_processes()
    
    service_names = {
        'options_analysis': 'Options Analysis Scheduler',
        'intraday_collection': 'Intraday Data Collection',
        'options_collection': 'Options Chain Collection'
    }
    
    all_healthy = True
    
    for service_key, service_name in service_names.items():
        pid = processes.get(service_key)
        is_healthy, uptime = check_process_health(pid)
        
        status_icon = "✅" if is_healthy else "❌"
        pid_info = f"PID: {pid}" if pid else "Not Running"
        uptime_info = f"Uptime: {str(uptime).split('.')[0]}" if uptime else ""
        
        print(f"{status_icon} {service_name:<25} | {pid_info:<15} | {uptime_info}")
        
        if not is_healthy:
            all_healthy = False
    
    print("=" * 60)
    
    if all_healthy:
        print("🎉 All services are running healthy!")
    else:
        print("⚠️  Some services are not running. Use start_all_background_services.py to start them.")
    
    print(f"\n📅 Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
