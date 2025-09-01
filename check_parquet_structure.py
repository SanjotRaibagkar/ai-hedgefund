#!/usr/bin/env python3
"""
Check parquet file structure
"""

import pandas as pd
from pathlib import Path

def check_parquet_structure():
    """Check the structure of a parquet file"""
    parquet_file = Path("data/options_parquet/20250829/NIFTY_20250829_152428.parquet")
    
    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
            print("✅ Parquet file loaded successfully!")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Check date column format
            print(f"\n📅 Date column info:")
            print(f"   Date column type: {type(df['date'].iloc[0])}")
            print(f"   Sample date values: {df['date'].head(3).tolist()}")
            print(f"   Unique dates: {df['date'].unique()}")
            
            # Check Fetch_Time column format
            print(f"\n⏰ Fetch_Time column info:")
            print(f"   Fetch_Time column type: {type(df['Fetch_Time'].iloc[0])}")
            print(f"   Sample Fetch_Time values: {df['Fetch_Time'].head(3).tolist()}")
            
            # Check Expiry_Date column format
            print(f"\n📆 Expiry_Date column info:")
            print(f"   Expiry_Date column type: {type(df['Expiry_Date'].iloc[0])}")
            print(f"   Sample Expiry_Date values: {df['Expiry_Date'].head(3).tolist()}")
            
            print(f"\n📅 Sample data:")
            print(df[['date', 'Fetch_Time', 'Expiry_Date', 'Strike_Price', 'CALLS_OI', 'PUTS_OI']].head(3))
            
            # Check for required columns
            required_cols = ['Fetch_Time', 'Expiry_Date', 'date', 'Strike_Price', 'CALLS_OI', 'PUTS_OI']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
            else:
                print(f"✅ All required columns present!")
                
        except Exception as e:
            print(f"❌ Error reading parquet file: {e}")
    else:
        print(f"❌ File not found: {parquet_file}")

if __name__ == "__main__":
    check_parquet_structure()
