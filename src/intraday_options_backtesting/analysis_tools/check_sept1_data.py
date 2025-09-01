#!/usr/bin/env python3
"""
Check September 1st, 2025 data to understand why prediction is wrong
"""

import pandas as pd
from pathlib import Path
import numpy as np

def check_sept1_data():
    print("🔍 CHECKING SEPTEMBER 1ST, 2025 DATA")
    print("=" * 80)
    
    # Path to September 1st data
    sept1_path = Path("../../../data/options_parquet/20250901")
    
    if not sept1_path.exists():
        print(f"❌ Path not found: {sept1_path}")
        return
    
    # List all NIFTY files
    nifty_files = list(sept1_path.glob("NIFTY_*.parquet"))
    print(f"📁 Found {len(nifty_files)} NIFTY files")
    
    if not nifty_files:
        print("❌ No NIFTY files found")
        return
    
    # Check a few key time files
    key_times = ["110018", "130014", "150032"]  # 11:00 AM, 1:00 PM, 3:00 PM
    
    for time_suffix in key_times:
        matching_files = [f for f in nifty_files if time_suffix in f.name]
        if matching_files:
            file_path = matching_files[0]
            print(f"\n⏰ Analyzing {file_path.name}")
            print("-" * 50)
            
            try:
                df = pd.read_parquet(file_path)
                print(f"   Records: {len(df)}")
                print(f"   Columns: {list(df.columns)}")
                
                # Check strike price range
                strikes = sorted(df['Strike_Price'].unique())
                print(f"   Strike Range: {strikes[0]:,} to {strikes[-1]:,}")
                print(f"   Mid Strike: {strikes[len(strikes)//2]:,}")
                
                # Check current expiry data
                current_expiry = df['Expiry_Date'].min()
                current_exp_data = df[df['Expiry_Date'] == current_expiry]
                
                print(f"   Current Expiry: {current_expiry}")
                print(f"   Current Expiry Records: {len(current_exp_data)}")
                
                # Calculate total OI by strike
                current_exp_data['Total_OI'] = current_exp_data['CALLS_OI'] + current_exp_data['PUTS_OI']
                max_oi_strike = current_exp_data.loc[current_exp_data['Total_OI'].idxmax(), 'Strike_Price']
                max_oi_value = current_exp_data['Total_OI'].max()
                
                print(f"   Max OI Strike: {max_oi_strike:,} (OI: {max_oi_value:,.0f})")
                
                # Check PCR
                total_call_oi = current_exp_data['CALLS_OI'].sum()
                total_put_oi = current_exp_data['PUTS_OI'].sum()
                pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                
                print(f"   Total Call OI: {total_call_oi:,.0f}")
                print(f"   Total Put OI: {total_put_oi:,.0f}")
                print(f"   PCR: {pcr:.3f}")
                
                # Check if there are any strikes around 24,500
                strikes_around_24500 = [s for s in strikes if 24400 <= s <= 24600]
                print(f"   Strikes around 24,500: {strikes_around_24500}")
                
                # Check data for strikes around 24,500
                if strikes_around_24500:
                    for strike in strikes_around_24500:
                        strike_data = current_exp_data[current_exp_data['Strike_Price'] == strike]
                        if len(strike_data) > 0:
                            call_oi = strike_data['CALLS_OI'].iloc[0]
                            put_oi = strike_data['PUTS_OI'].iloc[0]
                            print(f"     Strike {strike:,}: Call OI={call_oi:,.0f}, Put OI={put_oi:,.0f}")
                
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
                continue
    
    # Check if there are any files with different naming patterns
    print(f"\n🔍 CHECKING FOR OTHER FILE PATTERNS")
    print("-" * 50)
    
    # Look for any files that might contain actual spot price
    all_files = list(sept1_path.glob("*.parquet"))
    print(f"Total parquet files: {len(all_files)}")
    
    # Check first few files for any unusual patterns
    for i, file_path in enumerate(all_files[:5]):
        print(f"\nFile {i+1}: {file_path.name}")
        try:
            df = pd.read_parquet(file_path)
            print(f"   Records: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check if there's a spot price column
            if 'Spot_Price' in df.columns:
                spot_price = df['Spot_Price'].iloc[0]
                print(f"   ⭐ SPOT PRICE FOUND: {spot_price}")
            elif 'spot_price' in df.columns:
                spot_price = df['spot_price'].iloc[0]
                print(f"   ⭐ SPOT PRICE FOUND: {spot_price}")
            else:
                print(f"   No spot price column found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    check_sept1_data()
