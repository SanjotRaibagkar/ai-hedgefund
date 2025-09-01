#!/usr/bin/env python3
"""
Monitor Options CSV Updates
Simple script to monitor the options tracker CSV file for updates
"""

import os
import time
from datetime import datetime
import pandas as pd

def monitor_csv_updates():
    """Monitor the options tracker CSV file for updates"""
    
    csv_file = "results/options_tracker/option_tracker.csv"
    
    print("🔍 Monitoring Options CSV Updates")
    print("=" * 50)
    print(f"📁 File: {csv_file}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    last_modified = 0
    last_line_count = 0
    
    while True:
        try:
            # Check if file exists
            if not os.path.exists(csv_file):
                print(f"❌ CSV file not found: {csv_file}")
                time.sleep(30)
                continue
            
            # Get file modification time
            current_modified = os.path.getmtime(csv_file)
            
            # Get current line count
            with open(csv_file, 'r') as f:
                current_line_count = len(f.readlines())
            
            # Check if file was updated
            if current_modified != last_modified or current_line_count != last_line_count:
                print(f"\n📊 CSV Updated at {datetime.now().strftime('%H:%M:%S')}")
                print(f"   📈 Records: {current_line_count - 1} (excluding header)")
                
                # Read and display latest data
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        latest = df.iloc[-1]
                        print(f"   📊 Latest Analysis:")
                        print(f"      Index: {latest['index']}")
                        print(f"      Time: {latest['timestamp']}")
                        print(f"      Price: ₹{latest['current_spot_price']:,.0f}")
                        print(f"      PCR: {latest['pcr']:.2f}")
                        print(f"      Signal: {latest['signal']}")
                        print(f"      Confidence: {latest['confidence']}%")
                        print(f"      Trade: {latest['suggested_trade']}")
                except Exception as e:
                    print(f"   ❌ Error reading CSV: {e}")
                
                last_modified = current_modified
                last_line_count = current_line_count
            
            # Wait 30 seconds before next check
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_csv_updates()
