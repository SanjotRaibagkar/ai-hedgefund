#!/usr/bin/env python3
"""
Bhavcopy Data Downloader
Downloads real historical data using NSEUtility's bhavcopy method.
"""

import pandas as pd
import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta
from src.nsedata.NseUtility import NseUtils
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bhavcopy_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BhavcopyDownloader:
    """Download historical data using NSEUtility bhavcopy method."""
    
    def __init__(self, db_path: str = "data/bhavcopy_equity.db"):
        """Initialize the downloader."""
        self.db_path = db_path
        self.nse = NseUtils()
        self.progress_file = "bhavcopy_progress.json"
        
        # Initialize database
        self._init_database()
        
        # Load progress
        self.progress = self._load_progress()
        
    def _init_database(self):
        """Initialize the database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Create securities table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS securities (
                symbol TEXT PRIMARY KEY,
                company_name TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_updated TEXT,
                data_start_date TEXT,
                data_end_date TEXT,
                total_records INTEGER DEFAULT 0
            )
        ''')
        
        # Create price_data table
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
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON price_data(symbol)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _load_progress(self) -> dict:
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'completed_dates': [],
            'failed_dates': [],
            'total_dates': 0,
            'start_time': None,
            'last_updated': None
        }
    
    def _save_progress(self):
        """Save progress to file."""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_trading_dates(self, start_date: str = '2024-01-01', end_date: str = None) -> list:
        """Get list of trading dates."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_dates = []
        current_date = start
        
        while current_date <= end:
            # Skip weekends
            if current_date.weekday() < 5:
                trading_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        logger.info(f"Generated {len(trading_dates)} trading dates from {start_date} to {end_date}")
        return trading_dates
    
    def _get_bhavcopy_for_date(self, date: str) -> pd.DataFrame:
        """Get bhavcopy data for a specific date."""
        try:
            logger.info(f"Fetching bhavcopy for date: {date}")
            
            # Convert date to required format (DD-MM-YYYY)
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%d-%m-%Y')
            
            # Get bhavcopy data
            bhavcopy_data = self.nse.bhavcopy(formatted_date)
            
            if bhavcopy_data is not None and not bhavcopy_data.empty:
                logger.info(f"Successfully fetched bhavcopy for {date}: {len(bhavcopy_data)} records")
                return bhavcopy_data
            else:
                logger.warning(f"No bhavcopy data available for {date}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching bhavcopy for {date}: {e}")
            return pd.DataFrame()
    
    def _process_bhavcopy_data(self, bhavcopy_data: pd.DataFrame, date: str) -> list:
        """Process bhavcopy data and convert to our format."""
        processed_data = []
        
        try:
            # Check if required columns exist
            if 'SYMBOL' not in bhavcopy_data.columns:
                logger.warning(f"SYMBOL column not found in bhavcopy data for {date}")
                return []
            
            # Process each row
            for _, row in bhavcopy_data.iterrows():
                try:
                    symbol = str(row['SYMBOL']).strip()
                    
                    # Skip invalid symbols
                    if not symbol or symbol == 'nan' or symbol == '':
                        continue
                    
                    # Clean symbol name
                    symbol = symbol.replace('.NS', '').replace('.NSE', '').upper()
                    
                    # Extract price data (try different column names)
                    open_price = 0
                    high_price = 0
                    low_price = 0
                    close_price = 0
                    volume = 0
                    turnover = 0
                    
                    # Try to get price data from various possible column names
                    for col in ['OPEN', 'Open', 'open']:
                        if col in bhavcopy_data.columns and pd.notna(row[col]):
                            open_price = float(row[col])
                            break
                    
                    for col in ['HIGH', 'High', 'high']:
                        if col in bhavcopy_data.columns and pd.notna(row[col]):
                            high_price = float(row[col])
                            break
                    
                    for col in ['LOW', 'Low', 'low']:
                        if col in bhavcopy_data.columns and pd.notna(row[col]):
                            low_price = float(row[col])
                            break
                    
                    for col in ['CLOSE', 'Close', 'close']:
                        if col in bhavcopy_data.columns and pd.notna(row[col]):
                            close_price = float(row[col])
                            break
                    
                    for col in ['VOLUME', 'Volume', 'volume', 'TOTTRDQTY']:
                        if col in bhavcopy_data.columns and pd.notna(row[col]):
                            volume = int(row[col])
                            break
                    
                    for col in ['TURNOVER', 'Turnover', 'turnover', 'TOTTRDVAL']:
                        if col in bhavcopy_data.columns and pd.notna(row[col]):
                            turnover = float(row[col])
                            break
                    
                    # Skip if no meaningful data
                    if close_price <= 0:
                        continue
                    
                    processed_data.append({
                        'symbol': symbol,
                        'date': date,
                        'open_price': open_price,
                        'high_price': high_price,
                        'low_price': low_price,
                        'close_price': close_price,
                        'volume': volume,
                        'turnover': turnover
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing row in bhavcopy data for {date}: {e}")
                    continue
            
            logger.info(f"Processed {len(processed_data)} records for {date}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing bhavcopy data for {date}: {e}")
            return []
    
    def _store_bhavcopy_data(self, processed_data: list):
        """Store processed bhavcopy data in database."""
        if not processed_data:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get unique symbols from this batch
            symbols_in_batch = set(record['symbol'] for record in processed_data)
            
            # Update securities table
            for symbol in symbols_in_batch:
                symbol_data = [r for r in processed_data if r['symbol'] == symbol]
                
                if symbol_data:
                    # Calculate date range for this symbol
                    dates = [r['date'] for r in symbol_data]
                    start_date = min(dates)
                    end_date = max(dates)
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO securities 
                        (symbol, company_name, is_active, last_updated, data_start_date, data_end_date, total_records)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        symbol,  # Use symbol as company name for now
                        1,
                        datetime.now().isoformat(),
                        start_date,
                        end_date,
                        len(symbol_data)
                    ))
            
            # Insert price data
            for record in processed_data:
                conn.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, date, open_price, high_price, low_price, close_price, volume, turnover, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record['symbol'],
                    record['date'],
                    record['open_price'],
                    record['high_price'],
                    record['low_price'],
                    record['close_price'],
                    record['volume'],
                    record['turnover'],
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing bhavcopy data: {e}")
            raise
    
    def download_bhavcopy_data(self, start_date: str = '2024-01-01', end_date: str = None, max_dates: int = None):
        """Download bhavcopy data for all trading dates."""
        logger.info("Starting bhavcopy data download...")
        
        # Get trading dates
        trading_dates = self.get_trading_dates(start_date, end_date)
        
        if not trading_dates:
            logger.error("No trading dates to process!")
            return
        
        # Limit dates if specified
        if max_dates:
            trading_dates = trading_dates[:max_dates]
            logger.info(f"Limited to {len(trading_dates)} dates")
        
        # Update progress
        self.progress['total_dates'] = len(trading_dates)
        self.progress['start_time'] = datetime.now().isoformat()
        self._save_progress()
        
        # Download data
        completed = 0
        failed = 0
        
        logger.info(f"Starting download for {len(trading_dates)} dates...")
        
        for i, date in enumerate(trading_dates):
            try:
                # Check if already completed
                if date in self.progress['completed_dates']:
                    logger.info(f"Skipping {date} (already completed)")
                    completed += 1
                    continue
                
                # Get bhavcopy data
                bhavcopy_data = self._get_bhavcopy_for_date(date)
                
                if bhavcopy_data.empty:
                    logger.warning(f"No data available for {date}")
                    failed += 1
                    self.progress['failed_dates'].append({
                        'date': date,
                        'error': 'No data available',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
                
                # Process the data
                processed_data = self._process_bhavcopy_data(bhavcopy_data, date)
                
                if not processed_data:
                    logger.warning(f"No valid records found for {date}")
                    failed += 1
                    self.progress['failed_dates'].append({
                        'date': date,
                        'error': 'No valid records',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
                
                # Store the data
                self._store_bhavcopy_data(processed_data)
                
                # Update progress
                completed += 1
                self.progress['completed_dates'].append(date)
                self._save_progress()
                
                logger.info(f"✅ {date}: Downloaded {len(processed_data)} records")
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(trading_dates)} (Completed: {completed}, Failed: {failed})")
                
                # Rate limiting
                time.sleep(1)  # Conservative rate limiting for bhavcopy
                
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                failed += 1
                self.progress['failed_dates'].append({
                    'date': date,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                self._save_progress()
        
        # Final summary
        logger.info(f"Download completed! Total: {len(trading_dates)}, Completed: {completed}, Failed: {failed}")
        
        # Save final progress
        self.progress['completion_time'] = datetime.now().isoformat()
        self._save_progress()
        
        return {
            'total_dates': len(trading_dates),
            'completed': completed,
            'failed': failed,
            'success_rate': (completed / len(trading_dates)) * 100 if trading_dates else 0
        }
    
    def get_database_stats(self) -> dict:
        """Get statistics from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get securities count
            securities_count = conn.execute('SELECT COUNT(*) FROM securities').fetchone()[0]
            
            # Get price data count
            price_data_count = conn.execute('SELECT COUNT(*) FROM price_data').fetchone()[0]
            
            # Get unique symbols
            unique_symbols = conn.execute('SELECT COUNT(DISTINCT symbol) FROM price_data').fetchone()[0]
            
            # Get date range
            date_range = conn.execute('SELECT MIN(date), MAX(date) FROM price_data').fetchone()
            
            conn.close()
            
            return {
                'securities_count': securities_count,
                'price_data_count': price_data_count,
                'unique_symbols': unique_symbols,
                'date_range': date_range
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

def main():
    """Main function to run the bhavcopy downloader."""
    print("🚀 BHAVCOPY DATA DOWNLOADER")
    print("=" * 50)
    
    # Initialize downloader
    downloader = BhavcopyDownloader()
    
    try:
        # Start download
        print("\n🔄 Starting bhavcopy download...")
        print("📅 Date range: 2024-01-01 to today")
        print("⏱️ Estimated time: 2-3 hours")
        print("💡 Progress will be saved automatically")
        print()
        
        results = downloader.download_bhavcopy_data(start_date='2024-01-01')
        
        print("\n📊 DOWNLOAD RESULTS:")
        print("=" * 30)
        print(f"✅ Completed Dates: {results['completed']}")
        print(f"❌ Failed Dates: {results['failed']}")
        print(f"📋 Total Dates: {results['total_dates']}")
        print(f"📈 Success Rate: {results['success_rate']:.2f}%")
        
        # Get database stats
        stats = downloader.get_database_stats()
        if stats:
            print(f"\n📊 DATABASE STATISTICS:")
            print(f"   • Securities: {stats['securities_count']}")
            print(f"   • Price Records: {stats['price_data_count']}")
            print(f"   • Unique Symbols: {stats['unique_symbols']}")
            print(f"   • Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        
        print(f"\n📄 Files Created:")
        print(f"   • {downloader.db_path}")
        print(f"   • {downloader.progress_file}")
        print(f"   • bhavcopy_download.log")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        logger.error(f"Download failed: {e}")

if __name__ == "__main__":
    main()
