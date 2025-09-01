#!/usr/bin/env python3
"""
Options Data Table Display for Intraday Options Backtesting
Displays options chain data in clean, organized tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class OptionsDataTable:
    def __init__(self, parquet_folder_path):
        self.parquet_folder_path = Path(parquet_folder_path)
        
    def read_latest_data(self, date, symbol='NIFTY'):
        """Read the latest parquet file for a specific date and symbol"""
        date_folder = self.parquet_folder_path / date
        if not date_folder.exists():
            return None
            
        # Get the latest file for the symbol
        pattern = f"{symbol}_*.parquet"
        files = list(date_folder.glob(pattern))
        if not files:
            return None
            
        latest_file = sorted(files)[-1]
        print(f"📁 Reading latest file: {latest_file.name}")
        
        try:
            df = pd.read_parquet(latest_file)
            return df
        except Exception as e:
            print(f"Error reading {latest_file}: {e}")
            return None
    
    def display_options_table(self, df, symbol, expiry_date=None, strike_range=None):
        """Display options data in a clean table format"""
        if df is None or df.empty:
            print("No data to display")
            return
        
        # Filter by expiry date if specified
        if expiry_date:
            df = df[df['Expiry_Date'] == expiry_date]
            print(f"\n📅 Filtered by Expiry: {expiry_date}")
        
        # Filter by strike range if specified
        if strike_range:
            min_strike, max_strike = strike_range
            df = df[(df['Strike_Price'] >= min_strike) & (df['Strike_Price'] <= max_strike)]
            print(f"💰 Strike Range: {min_strike:,} - {max_strike:,}")
        
        if df.empty:
            print("No data matches the specified filters")
            return
        
        # Sort by strike price
        df = df.sort_values('Strike_Price')
        
        print(f"\n{'='*120}")
        print(f"OPTIONS CHAIN TABLE - {symbol}")
        print(f"{'='*120}")
        
        # Select columns to display
        display_cols = [
            'Strike_Price', 'Expiry_Date',
            'CALLS_OI', 'CALLS_Chng_in_OI', 'CALLS_Volume', 'CALLS_IV', 'CALLS_LTP',
            'PUTS_OI', 'PUTS_Chng_in_OI', 'PUTS_Volume', 'PUTS_IV', 'PUTS_LTP'
        ]
        
        # Filter available columns
        available_cols = [col for col in display_cols if col in df.columns]
        df_display = df[available_cols].copy()
        
        # Format numeric columns
        numeric_cols = ['CALLS_OI', 'CALLS_Chng_in_OI', 'CALLS_Volume', 'CALLS_IV', 'CALLS_LTP',
                       'PUTS_OI', 'PUTS_Chng_in_OI', 'PUTS_Volume', 'PUTS_IV', 'PUTS_LTP']
        
        for col in numeric_cols:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                # Format based on column type
                if 'OI' in col or 'Volume' in col:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                elif 'IV' in col:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                elif 'LTP' in col:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                else:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
        
        # Format strike price
        if 'Strike_Price' in df_display.columns:
            df_display['Strike_Price'] = df_display['Strike_Price'].apply(lambda x: f"{x:,.0f}")
        
        # Display the table
        print(df_display.to_string(index=False))
        
        # Show summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   Total strikes: {len(df)}")
        if 'Strike_Price' in df.columns:
            print(f"   Strike range: {df['Strike_Price'].min():,.0f} - {df['Strike_Price'].max():,.0f}")
        
        # Calculate PCR (Put-Call Ratio)
        if 'CALLS_OI' in df.columns and 'PUTS_OI' in df.columns:
            calls_oi = pd.to_numeric(df['CALLS_OI'], errors='coerce').sum()
            puts_oi = pd.to_numeric(df['PUTS_OI'], errors='coerce').sum()
            if calls_oi > 0:
                pcr = puts_oi / calls_oi
                print(f"   Put-Call Ratio (PCR): {pcr:.3f}")
        
        return df_display
    
    def show_atm_options(self, df, symbol, expiry_date=None):
        """Show At-The-Money options around current market price"""
        if df is None or df.empty:
            return
        
        # Filter by expiry date if specified
        if expiry_date:
            df = df[df['Expiry_Date'] == expiry_date]
        
        if df.empty:
            return
        
        # Try to find current market price (approximate from strike prices)
        # For NIFTY, typically around 19,000-20,000
        # For BANKNIFTY, typically around 45,000-50,000
        
        if symbol == 'NIFTY':
            target_price = 19500  # Approximate current NIFTY price
        elif symbol == 'BANKNIFTY':
            target_price = 47000  # Approximate current BANKNIFTY price
        else:
            target_price = df['Strike_Price'].median()
        
        # Find strikes around target price (±500 for NIFTY, ±1000 for BANKNIFTY)
        if symbol == 'NIFTY':
            range_size = 500
        else:
            range_size = 1000
            
        atm_df = df[
            (df['Strike_Price'] >= target_price - range_size) & 
            (df['Strike_Price'] <= target_price + range_size)
        ].copy()
        
        if not atm_df.empty:
            print(f"\n🎯 ATM Options (Around {target_price:,}):")
            print(f"{'='*80}")
            
            # Sort by strike price
            atm_df = atm_df.sort_values('Strike_Price')
            
            # Select key columns for ATM display
            atm_cols = ['Strike_Price', 'CALLS_OI', 'CALLS_LTP', 'PUTS_OI', 'PUTS_LTP']
            available_atm_cols = [col for col in atm_cols if col in atm_df.columns]
            
            atm_display = atm_df[available_atm_cols].copy()
            
            # Format numeric columns
            for col in available_atm_cols:
                if col == 'Strike_Price':
                    atm_display[col] = atm_display[col].apply(lambda x: f"{x:,.0f}")
                elif 'OI' in col:
                    atm_display[col] = pd.to_numeric(atm_display[col], errors='coerce').apply(
                        lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
                    )
                elif 'LTP' in col:
                    atm_display[col] = pd.to_numeric(atm_display[col], errors='coerce').apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                    )
            
            print(atm_display.to_string(index=False))
    
    def analyze_intraday_movement(self, date, symbol='NIFTY'):
        """Analyze intraday movement by comparing first and last files"""
        date_folder = self.parquet_folder_path / date
        if not date_folder.exists():
            return
        
        # Get first and last files
        pattern = f"{symbol}_*.parquet"
        files = sorted(list(date_folder.glob(pattern)))
        
        if len(files) < 2:
            print(f"Not enough files for {symbol} on {date}")
            return
        
        first_file = files[0]
        last_file = files[-1]
        
        print(f"\n📈 Intraday Movement Analysis - {symbol}")
        print(f"{'='*80}")
        print(f"First file: {first_file.name}")
        print(f"Last file:  {last_file.name}")
        
        # Read both files
        df_first = pd.read_parquet(first_file)
        df_last = pd.read_parquet(last_file)
        
        # Find common strikes and expiry dates for comparison
        common_strikes = set(df_first['Strike_Price']) & set(df_last['Strike_Price'])
        common_expiries = set(df_first['Expiry_Date']) & set(df_last['Expiry_Date'])
        
        print(f"Common strikes: {len(common_strikes)}")
        print(f"Common expiries: {len(common_expiries)}")
        
        # Show sample comparison for a few strikes
        sample_strikes = sorted(list(common_strikes))[:5]
        
        for strike in sample_strikes:
            print(f"\n💰 Strike: {strike:,}")
            
            # Get data for this strike from both files
            first_data = df_first[df_first['Strike_Price'] == strike].iloc[0] if len(df_first[df_first['Strike_Price'] == strike]) > 0 else None
            last_data = df_last[df_last['Strike_Price'] == strike].iloc[0] if len(df_last[df_last['Strike_Price'] == strike]) > 0 else None
            
            if first_data is not None and last_data is not None:
                print(f"   CALLS OI: {first_data.get('CALLS_OI', 'N/A')} → {last_data.get('CALLS_OI', 'N/A')}")
                print(f"   CALLS LTP: {first_data.get('CALLS_LTP', 'N/A')} → {last_data.get('CALLS_LTP', 'N/A')}")
                print(f"   PUTS OI: {first_data.get('PUTS_OI', 'N/A')} → {last_data.get('PUTS_OI', 'N/A')}")
                print(f"   PUTS LTP: {first_data.get('PUTS_LTP', 'N/A')} → {last_data.get('PUTS_LTP', 'N/A')}")

def main():
    """Main function to run the options data table display"""
    parquet_folder = "data/options_parquet"
    analyzer = OptionsDataTable(parquet_folder)
    
    # Target date
    target_date = "20250829"
    
    print(f"🎯 Analyzing Options Data for {target_date}")
    print(f"{'='*80}")
    
    # Analyze NIFTY data
    print("\n📊 NIFTY ANALYSIS:")
    nifty_df = analyzer.read_latest_data(target_date, 'NIFTY')
    
    if nifty_df is not None:
        # Show main options table for September expiry
        analyzer.display_options_table(nifty_df, 'NIFTY', '30-Sep-2025')
        
        # Show ATM options
        analyzer.show_atm_options(nifty_df, 'NIFTY', '30-Sep-2025')
        
        # Analyze intraday movement
        analyzer.analyze_intraday_movement(target_date, 'NIFTY')
    
    print("\n" + "="*120)
    
    # Analyze BANKNIFTY data
    print("\n📊 BANKNIFTY ANALYSIS:")
    banknifty_df = analyzer.read_latest_data(target_date, 'BANKNIFTY')
    
    if banknifty_df is not None:
        # Show main options table for September expiry
        analyzer.display_options_table(banknifty_df, 'BANKNIFTY', '30-Sep-2025')
        
        # Show ATM options
        analyzer.show_atm_options(banknifty_df, 'BANKNIFTY', '30-Sep-2025')
        
        # Analyze intraday movement
        analyzer.analyze_intraday_movement(target_date, 'BANKNIFTY')

if __name__ == "__main__":
    main()
