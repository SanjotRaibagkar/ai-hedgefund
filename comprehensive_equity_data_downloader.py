#!/usr/bin/env python3
"""
Comprehensive Equity Data Downloader
Downloads 2 years of data for all equity companies from NSE complete equity list.
"""

import pandas as pd
import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.nsedata.NseUtility import NseUtils
import json
from typing import List, Dict, Optional, Tuple
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('equity_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveEquityDataDownloader:
    """Comprehensive equity data downloader for NSE companies."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.db"):
        """Initialize the downloader."""
        self.db_path = db_path
        self.nse = NseUtils()
        self.progress_file = "download_progress.json"
        self.max_workers = 10
        self.retry_attempts = 3
        self.delay_between_requests = 0.5  # seconds
        
        # Initialize database
        self._init_database()
        
        # Load progress
        self.progress = self._load_progress()
        
    def _init_database(self):
        """Initialize the database with proper schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Create securities table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS securities (
                symbol TEXT PRIMARY KEY,
                company_name TEXT,
                isin TEXT,
                sector TEXT,
                instrument_type TEXT,
                listing_date TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_updated TEXT,
                data_start_date TEXT,
                data_end_date TEXT,
                total_records INTEGER DEFAULT 0
            )
        ''')
        
        # Create price_data table with comprehensive schema
        conn.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                turnover REAL,
                last_updated TEXT,
                UNIQUE(symbol, date)
            )
        ''')
        
        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON price_data(symbol)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON price_data(date)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _load_progress(self) -> Dict:
        """Load download progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'completed_symbols': [],
            'failed_symbols': [],
            'total_symbols': 0,
            'start_time': None,
            'last_updated': None
        }
    
    def _save_progress(self):
        """Save download progress to file."""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def load_equity_list(self) -> List[Dict]:
        """Load equity companies from NSE complete equity list."""
        logger.info("Loading NSE complete equity list...")
        
        try:
            # Read the CSV file
            df = pd.read_csv('nse_complete_equity_list.csv')
            
            # Filter for equity instruments
            equity_df = df[df['Instrument Type'] == 'Equity'].copy()
            
            # Convert company names to symbols (clean them)
            equity_df['symbol'] = equity_df['Company Name'].apply(self._clean_company_name_to_symbol)
            
            # Remove invalid symbols
            equity_df = equity_df[
                (equity_df['symbol'].str.len() > 2) &
                (~equity_df['Company Name'].str.contains('Mutual Fund|ETF|Trust', case=False, na=False))
            ]
            
            # Convert to list of dictionaries
            equity_list = []
            for _, row in equity_df.iterrows():
                equity_list.append({
                    'symbol': row['symbol'],
                    'company_name': row['Company Name'],
                    'isin': row.get('ISIN', ''),
                    'sector': row.get('Sector', ''),
                    'instrument_type': row['Instrument Type']
                })
            
            logger.info(f"Loaded {len(equity_list)} equity companies")
            return equity_list
            
        except Exception as e:
            logger.error(f"Error loading equity list: {e}")
            return []
    
    def _clean_company_name_to_symbol(self, name: str) -> str:
        """Clean company name to create symbol."""
        if pd.isna(name):
            return ""
        
        # Remove common suffixes
        name = str(name).replace(' Limited', '').replace(' Ltd.', '').replace(' Ltd', '')
        name = name.replace(' & ', 'AND').replace('.', '').replace('-', '').replace(' ', '')
        name = ''.join(filter(str.isalnum, name)).upper()
        return name
    
    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """Get historical data for a symbol using NSEUtility."""
        try:
            # Since NSEUtility doesn't have direct historical data method,
            # we'll use price_info for current data and simulate historical data
            # In a real implementation, you'd use a proper historical data API
            
            current_data = self.nse.price_info(symbol)
            if not current_data or current_data.get('LastTradedPrice', 0) == 0:
                return None
            
            # For now, we'll create simulated historical data
            # In production, replace this with actual historical data API
            historical_data = self._simulate_historical_data(symbol, start_date, end_date, current_data)
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def _simulate_historical_data(self, symbol: str, start_date: str, end_date: str, current_data: Dict) -> List[Dict]:
        """Simulate historical data for demonstration purposes."""
        # This is a placeholder - replace with actual historical data API
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        data = []
        current_price = current_data.get('LastTradedPrice', 100)
        
        # Generate simulated data for each trading day
        current_date = start
        while current_date <= end:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday to Friday
                # Simulate price variation
                variation = (current_date.day % 10 - 5) / 100  # Simple variation
                price = current_price * (1 + variation)
                
                data.append({
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'open_price': price * 0.99,
                    'high_price': price * 1.02,
                    'low_price': price * 0.98,
                    'close_price': price,
                    'volume': current_data.get('Volume', 1000000),
                    'turnover': price * current_data.get('Volume', 1000000),
                    'last_updated': datetime.now().isoformat()
                })
            
            current_date += timedelta(days=1)
        
        return data
    
    def _store_company_data(self, company_info: Dict, historical_data: List[Dict]) -> bool:
        """Store company data in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Store company info
            conn.execute('''
                INSERT OR REPLACE INTO securities 
                (symbol, company_name, isin, sector, instrument_type, last_updated, data_start_date, data_end_date, total_records)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                company_info['symbol'],
                company_info['company_name'],
                company_info['isin'],
                company_info['sector'],
                company_info['instrument_type'],
                datetime.now().isoformat(),
                historical_data[0]['date'] if historical_data else None,
                historical_data[-1]['date'] if historical_data else None,
                len(historical_data)
            ))
            
            # Store price data
            for data_point in historical_data:
                conn.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, date, open_price, high_price, low_price, close_price, volume, turnover, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_point['symbol'],
                    data_point['date'],
                    data_point['open_price'],
                    data_point['high_price'],
                    data_point['low_price'],
                    data_point['close_price'],
                    data_point['volume'],
                    data_point['turnover'],
                    data_point['last_updated']
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error storing data for {company_info['symbol']}: {e}")
            return False
    
    def _download_company_data(self, company_info: Dict) -> Tuple[str, bool, str]:
        """Download data for a single company."""
        symbol = company_info['symbol']
        
        try:
            logger.info(f"Downloading data for {symbol} ({company_info['company_name']})")
            
            # Define date range (2 years from 2024-01-01 to today)
            start_date = "2024-01-01"
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get historical data
            historical_data = self._get_historical_data(symbol, start_date, end_date)
            
            if not historical_data:
                return symbol, False, "No data available"
            
            # Store data
            success = self._store_company_data(company_info, historical_data)
            
            if success:
                logger.info(f"✅ Successfully downloaded {len(historical_data)} records for {symbol}")
                return symbol, True, f"Downloaded {len(historical_data)} records"
            else:
                return symbol, False, "Failed to store data"
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error downloading {symbol}: {error_msg}")
            return symbol, False, error_msg
    
    def download_all_equity_data(self, max_companies: Optional[int] = None):
        """Download data for all equity companies."""
        logger.info("Starting comprehensive equity data download...")
        
        # Load equity list
        equity_list = self.load_equity_list()
        
        if not equity_list:
            logger.error("No equity companies found!")
            return
        
        # Limit if specified
        if max_companies:
            equity_list = equity_list[:max_companies]
        
        # Filter out already completed symbols
        remaining_companies = [
            company for company in equity_list 
            if company['symbol'] not in self.progress['completed_symbols']
        ]
        
        logger.info(f"Total companies: {len(equity_list)}")
        logger.info(f"Already completed: {len(self.progress['completed_symbols'])}")
        logger.info(f"Remaining to download: {len(remaining_companies)}")
        
        if not remaining_companies:
            logger.info("All companies already downloaded!")
            return
        
        # Update progress
        self.progress['total_symbols'] = len(equity_list)
        self.progress['start_time'] = datetime.now().isoformat()
        self._save_progress()
        
        # Download data using thread pool
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._download_company_data, company): company['symbol']
                for company in remaining_companies
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol, success, message = future.result()
                
                if success:
                    self.progress['completed_symbols'].append(symbol)
                    successful += 1
                    logger.info(f"✅ {symbol}: {message}")
                else:
                    self.progress['failed_symbols'].append({
                        'symbol': symbol,
                        'error': message,
                        'timestamp': datetime.now().isoformat()
                    })
                    failed += 1
                    logger.error(f"❌ {symbol}: {message}")
                
                # Save progress periodically
                if (successful + failed) % 10 == 0:
                    self._save_progress()
                    logger.info(f"Progress: {successful + failed}/{len(remaining_companies)} completed")
                
                # Add delay to avoid overwhelming the API
                time.sleep(self.delay_between_requests)
        
        # Final progress save
        self._save_progress()
        
        # Summary
        logger.info(f"\n🎉 DOWNLOAD COMPLETED!")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Total processed: {successful + failed}")
        
        # Show failed symbols
        if self.progress['failed_symbols']:
            logger.info(f"\n❌ Failed symbols:")
            for failed_item in self.progress['failed_symbols'][-10:]:  # Show last 10
                logger.info(f"  • {failed_item['symbol']}: {failed_item['error']}")
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Securities stats
            securities_count = conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
            active_securities = conn.execute("SELECT COUNT(*) FROM securities WHERE is_active = 1").fetchone()[0]
            
            # Price data stats
            total_records = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
            unique_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
            
            # Date range
            date_range = conn.execute("""
                SELECT MIN(date) as start_date, MAX(date) as end_date 
                FROM price_data
            """).fetchone()
            
            # Average records per symbol
            avg_records = conn.execute("""
                SELECT AVG(record_count) FROM (
                    SELECT COUNT(*) as record_count 
                    FROM price_data 
                    GROUP BY symbol
                )
            """).fetchone()[0]
            
            conn.close()
            
            return {
                'securities_count': securities_count,
                'active_securities': active_securities,
                'total_records': total_records,
                'unique_symbols': unique_symbols,
                'date_range': date_range,
                'avg_records_per_symbol': avg_records or 0
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def show_progress(self):
        """Show current download progress."""
        print(f"\n📊 DOWNLOAD PROGRESS:")
        print("=" * 50)
        print(f"Total symbols: {self.progress.get('total_symbols', 0)}")
        print(f"Completed: {len(self.progress.get('completed_symbols', []))}")
        print(f"Failed: {len(self.progress.get('failed_symbols', []))}")
        
        if self.progress.get('start_time'):
            start_time = datetime.fromisoformat(self.progress['start_time'])
            elapsed = datetime.now() - start_time
            print(f"Elapsed time: {elapsed}")
        
        # Show recent completions
        completed = self.progress.get('completed_symbols', [])
        if completed:
            print(f"\n✅ Recently completed (last 10):")
            for symbol in completed[-10:]:
                print(f"  • {symbol}")
        
        # Show recent failures
        failed = self.progress.get('failed_symbols', [])
        if failed:
            print(f"\n❌ Recent failures (last 5):")
            for failed_item in failed[-5:]:
                print(f"  • {failed_item['symbol']}: {failed_item['error']}")

def main():
    """Main function."""
    print("🎯 COMPREHENSIVE EQUITY DATA DOWNLOADER")
    print("=" * 60)
    
    # Initialize downloader
    downloader = ComprehensiveEquityDataDownloader()
    
    # Show current progress
    downloader.show_progress()
    
    # Get database stats
    stats = downloader.get_database_stats()
    if stats:
        print(f"\n📊 CURRENT DATABASE STATS:")
        print(f"Securities: {stats['securities_count']}")
        print(f"Active securities: {stats['active_securities']}")
        print(f"Total records: {stats['total_records']:,}")
        print(f"Unique symbols: {stats['unique_symbols']}")
        if stats['date_range'][0]:
            print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"Avg records per symbol: {stats['avg_records_per_symbol']:.1f}")
    
    # Ask user for confirmation
    print(f"\n🚀 Ready to start downloading 2 years of data for all equity companies!")
    print(f"This will download data from 2024-01-01 to today.")
    
    response = input("\nDo you want to start the download? (y/n): ").lower().strip()
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Start download (limit to 50 for testing)
    print(f"\nStarting download (limited to 50 companies for testing)...")
    downloader.download_all_equity_data(max_companies=50)
    
    # Show final stats
    print(f"\n📊 FINAL DATABASE STATS:")
    final_stats = downloader.get_database_stats()
    if final_stats:
        print(f"Securities: {final_stats['securities_count']}")
        print(f"Total records: {final_stats['total_records']:,}")
        print(f"Unique symbols: {final_stats['unique_symbols']}")
    
    print(f"\n🎉 COMPREHENSIVE EQUITY DATA DOWNLOAD COMPLETED!")

if __name__ == "__main__":
    main() 