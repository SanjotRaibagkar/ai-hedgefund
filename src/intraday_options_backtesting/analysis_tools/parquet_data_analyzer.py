#!/usr/bin/env python3
"""
Parquet Data Analyzer for Intraday Options Backtesting
Analyzes and displays options chain data from parquet files
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import glob

class ParquetDataAnalyzer:
    def __init__(self, parquet_folder_path):
        self.parquet_folder_path = Path(parquet_folder_path)
        
    def get_available_dates(self):
        """Get all available dates in the parquet folder"""
        dates = []
        for folder in self.parquet_folder_path.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                dates.append(folder.name)
        return sorted(dates)
    
    def get_parquet_files(self, date):
        """Get all parquet files for a specific date"""
        date_folder = self.parquet_folder_path / date
        if not date_folder.exists():
            return []
        
        nifty_files = list(date_folder.glob("NIFTY_*.parquet"))
        banknifty_files = list(date_folder.glob("BANKNIFTY_*.parquet"))
        
        return {
            'NIFTY': sorted(nifty_files),
            'BANKNIFTY': sorted(banknifty_files)
        }
    
    def read_parquet_file(self, file_path):
        """Read a single parquet file and return DataFrame"""
        try:
            df = pd.read_parquet(file_path)
            # Add file info
            df['source_file'] = file_path.name
            df['file_timestamp'] = file_path.stem.split('_')[-1]
            return df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def analyze_symbol_data(self, date, symbol='NIFTY'):
        """Analyze data for a specific symbol on a specific date"""
        files = self.get_parquet_files(date)
        if symbol not in files or not files[symbol]:
            print(f"No {symbol} files found for date {date}")
            return None
        
        # Read first and last file to see data range
        first_file = files[symbol][0]
        last_file = files[symbol][-1]
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR {symbol} - DATE: {date}")
        print(f"{'='*80}")
        
        # Read first file
        df_first = self.read_parquet_file(first_file)
        if df_first is None:
            return None
            
        # Read last file
        df_last = self.read_parquet_file(last_file)
        if df_last is None:
            return None
        
        print(f"📊 Data Summary:")
        print(f"   First file: {first_file.name} ({len(df_first)} records)")
        print(f"   Last file:  {last_file.name} ({len(df_last)} records)")
        print(f"   Total files: {len(files[symbol])}")
        
        # Show data structure
        print(f"\n📋 Data Structure:")
        print(f"   Columns: {len(df_first.columns)}")
        print(f"   Sample columns: {df_first.columns[:10].tolist()}")
        
        # Show sample data
        print(f"\n📈 Sample Data (First 5 rows from {first_file.name}):")
        display_cols = ['Fetch_Time', 'Symbol', 'Strike_Price', 'CALLS_OI', 'CALLS_LTP', 
                       'PUTS_OI', 'PUTS_LTP', 'Expiry_Date']
        available_cols = [col for col in display_cols if col in df_first.columns]
        
        if available_cols:
            sample_df = df_first[available_cols].head()
            print(sample_df.to_string(index=False))
        
        # Show time range
        if 'Fetch_Time' in df_first.columns:
            print(f"\n⏰ Time Range:")
            print(f"   First: {df_first['Fetch_Time'].min()}")
            print(f"   Last:  {df_first['Fetch_Time'].max()}")
        
        # Show strike price range
        if 'Strike_Price' in df_first.columns:
            print(f"\n💰 Strike Price Range:")
            print(f"   Min: {df_first['Strike_Price'].min():,.0f}")
            print(f"   Max: {df_first['Strike_Price'].max():,.0f}")
            print(f"   Unique strikes: {df_first['Strike_Price'].nunique()}")
        
        # Show expiry dates
        if 'Expiry_Date' in df_first.columns:
            print(f"\n📅 Expiry Dates:")
            expiry_dates = df_first['Expiry_Date'].unique()
            for exp_date in sorted(expiry_dates):
                count = len(df_first[df_first['Expiry_Date'] == exp_date])
                print(f"   {exp_date}: {count} records")
        
        return df_first
    
    def show_file_timeline(self, date):
        """Show timeline of files for a specific date"""
        files = self.get_parquet_files(date)
        
        print(f"\n{'='*80}")
        print(f"FILE TIMELINE FOR DATE: {date}")
        print(f"{'='*80}")
        
        for symbol in ['NIFTY', 'BANKNIFTY']:
            if symbol in files and files[symbol]:
                print(f"\n📁 {symbol} Files ({len(files[symbol])} total):")
                
                # Group by hour for better readability
                hourly_files = {}
                for file_path in files[symbol]:
                    timestamp = file_path.stem.split('_')[-1]
                    hour = timestamp[:2]
                    if hour not in hourly_files:
                        hourly_files[hour] = []
                    hourly_files[hour].append(timestamp)
                
                for hour in sorted(hourly_files.keys()):
                    timestamps = hourly_files[hour]
                    print(f"   {hour}:00 - {len(timestamps)} files")
                    if len(timestamps) <= 5:  # Show all if few files
                        for ts in timestamps:
                            print(f"     {ts}")
                    else:  # Show range if many files
                        print(f"     {timestamps[0]} to {timestamps[-1]}")
    
    def analyze_all_data(self, date):
        """Analyze all data for a specific date"""
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE ANALYSIS FOR DATE: {date}")
        print(f"{'='*100}")
        
        # Show file timeline
        self.show_file_timeline(date)
        
        # Analyze NIFTY data
        nifty_df = self.analyze_symbol_data(date, 'NIFTY')
        
        # Analyze BANKNIFTY data
        banknifty_df = self.analyze_symbol_data(date, 'BANKNIFTY')
        
        return nifty_df, banknifty_df

def main():
    """Main function to run the analyzer"""
    # Path to options parquet folder
    parquet_folder = "data/options_parquet"
    
    analyzer = ParquetDataAnalyzer(parquet_folder)
    
    # Get available dates
    available_dates = analyzer.get_available_dates()
    print(f"📅 Available dates: {available_dates}")
    
    if not available_dates:
        print("No parquet data found!")
        return
    
    # Analyze the most recent date (August 29, 2025)
    target_date = "20250829"
    if target_date in available_dates:
        print(f"\n🎯 Analyzing data for {target_date}...")
        nifty_df, banknifty_df = analyzer.analyze_all_data(target_date)
    else:
        print(f"Target date {target_date} not found. Available: {available_dates}")
        # Use the first available date
        target_date = available_dates[0]
        print(f"Using available date: {target_date}")
        nifty_df, banknifty_df = analyzer.analyze_all_data(target_date)

if __name__ == "__main__":
    main()
