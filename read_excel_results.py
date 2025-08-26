#!/usr/bin/env python3
"""
Read Excel Results from FNO ML Strategy Analysis
Display strong file names and analysis results.
"""

import pandas as pd
import os
from datetime import datetime

def find_latest_excel_file():
    """Find the most recent Excel file in the data directory."""
    data_dir = "data"
    excel_files = [f for f in os.listdir(data_dir) if f.startswith("fno_current_analysis_") and f.endswith(".xlsx")]
    
    if not excel_files:
        print("No Excel files found in data directory")
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(excel_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
    return os.path.join(data_dir, latest_file)

def read_and_display_results():
    """Read the Excel file and display strong file names."""
    try:
        excel_file = find_latest_excel_file()
        if not excel_file:
            return
        
        print(f"📊 Reading Excel file: {excel_file}")
        print("=" * 80)
        
        # Read all sheets
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        
        for sheet_name, df in excel_data.items():
            print(f"\n📋 Sheet: {sheet_name}")
            print("-" * 50)
            
            if df.empty:
                print("No data in this sheet")
                continue
            
            print(f"Total symbols: {len(df)}")
            
            # Display strong signals (high confidence)
            if 'Confidence' in df.columns:
                strong_signals = df[df['Confidence'] >= 0.8].copy()
                print(f"\n🔥 STRONG SIGNALS (Confidence >= 0.8): {len(strong_signals)}")
                if not strong_signals.empty:
                    strong_signals = strong_signals.sort_values('Expected_Return_1d', ascending=False)
                    for _, row in strong_signals.head(10).iterrows():
                        signal_emoji = "🟢" if row['Signal'] == 'BUY' else "🔴" if row['Signal'] == 'SELL' else "🟡"
                        print(f"  {signal_emoji} {row['Symbol']}: {row['Expected_Return_1d']:.2f}% (conf: {row['Confidence']:.3f})")
            
            # Display top performers by expected return
            if 'Expected_Return_1d' in df.columns:
                print(f"\n📈 TOP 10 BY EXPECTED RETURN:")
                top_performers = df.sort_values('Expected_Return_1d', ascending=False).head(10)
                for _, row in top_performers.iterrows():
                    signal_emoji = "🟢" if row['Signal'] == 'BUY' else "🔴" if row['Signal'] == 'SELL' else "🟡"
                    print(f"  {signal_emoji} {row['Symbol']}: {row['Expected_Return_1d']:.2f}% (conf: {row['Confidence']:.3f})")
            
            # Display quartile breakdown
            if 'Expected_Return_1d' in df.columns:
                print(f"\n📊 QUARTILE BREAKDOWN:")
                quartiles = df['Expected_Return_1d'].describe()
                print(f"  Q1 (25%): {quartiles['25%']:.2f}%")
                print(f"  Q2 (50%): {quartiles['50%']:.2f}%")
                print(f"  Q3 (75%): {quartiles['75%']:.2f}%")
                print(f"  Max: {quartiles['max']:.2f}%")
                print(f"  Min: {quartiles['min']:.2f}%")
            
            # Display signal distribution
            if 'Signal' in df.columns:
                print(f"\n📊 SIGNAL DISTRIBUTION:")
                signal_counts = df['Signal'].value_counts()
                for signal, count in signal_counts.items():
                    signal_emoji = "🟢" if signal == 'BUY' else "🔴" if signal == 'SELL' else "🟡"
                    print(f"  {signal_emoji} {signal}: {count}")
            
            print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    read_and_display_results()
