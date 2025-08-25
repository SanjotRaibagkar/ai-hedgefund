#!/usr/bin/env python3
"""
EOD Extra Data Downloader
Downloads additional EOD data from NSE using NSE Utility methods.
"""

import pandas as pd
import duckdb
import os
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.nsedata.NseUtility import NseUtils
import json
from typing import List, Dict, Optional, Tuple, Any
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eod_extra_data_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EODExtraDataDownloader:
    """Downloads EOD extra data from NSE using NSE Utility methods."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        """Initialize the EOD extra data downloader."""
        self.db_path = db_path
        self.nse = NseUtils()
        self.progress_file = "eod_extra_download_progress.json"
        self.max_workers = 5
        self.retry_attempts = 3
        self.delay_between_requests = 0.5
        
        # Initialize database
        self._init_database()
        
        # Load progress
        self.progress = self._load_progress()
        
    def _init_database(self):
        """Initialize DuckDB with EOD extra data tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to DuckDB
        self.conn = duckdb.connect(self.db_path)
        
        # Create FNO Bhav Copy table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS fno_bhav_copy (
                SYMBOL VARCHAR,
                EXPIRY_DT VARCHAR,
                STRIKE_PRICE DOUBLE,
                OPTION_TYP VARCHAR,
                OPEN DOUBLE,
                HIGH DOUBLE,
                LOW DOUBLE,
                CLOSE DOUBLE,
                SETTLE_PR DOUBLE,
                CONTRACTS BIGINT,
                VAL_INLAKH DOUBLE,
                OPEN_INT BIGINT,
                CHG_IN_OI BIGINT,
                TRADE_DATE DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (SYMBOL, EXPIRY_DT, STRIKE_PRICE, OPTION_TYP, TRADE_DATE)
            )
        ''')
        
        # Create Equity Bhav Copy with Delivery table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS equity_bhav_copy_delivery (
                SYMBOL VARCHAR,
                SERIES VARCHAR,
                DATE1 VARCHAR,
                PREV_CLOSE DOUBLE,
                OPEN_PRICE DOUBLE,
                HIGH_PRICE DOUBLE,
                LOW_PRICE DOUBLE,
                LAST_PRICE DOUBLE,
                CLOSE_PRICE DOUBLE,
                AVG_PRICE DOUBLE,
                TTL_TRD_QNTY BIGINT,
                TURNOVER_LACS DOUBLE,
                NO_OF_TRADES BIGINT,
                DELIV_QTY BIGINT,
                DELIV_PER DOUBLE,
                TRADE_DATE DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (SYMBOL, SERIES, TRADE_DATE)
            )
        ''')
        
        # Create Bhav Copy Indices table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS bhav_copy_indices (
                Index_Name VARCHAR,
                Index_Date VARCHAR,
                Open_Index_Value DOUBLE,
                High_Index_Value DOUBLE,
                Low_Index_Value DOUBLE,
                Closing_Index_Value DOUBLE,
                Points_Change DOUBLE,
                Change_Percent DOUBLE,
                Volume DOUBLE,
                Turnover_Crs DOUBLE,
                P/E DOUBLE,
                P/B DOUBLE,
                Div_Yield DOUBLE,
                TRADE_DATE DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (Index_Name, TRADE_DATE)
            )
        ''')
        
        # Create FII DII Activity table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS fii_dii_activity (
                category VARCHAR,
                buyValue DOUBLE,
                buyQuantity BIGINT,
                sellValue DOUBLE,
                sellQuantity BIGINT,
                netValue DOUBLE,
                netQuantity BIGINT,
                activity_date DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (category, activity_date)
            )
        ''')
        
        # Create indexes for better performance
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_fno_symbol_date ON fno_bhav_copy(SYMBOL, TRADE_DATE)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_equity_symbol_date ON equity_bhav_copy_delivery(SYMBOL, TRADE_DATE)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_indices_name_date ON bhav_copy_indices(Index_Name, TRADE_DATE)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_fii_category_date ON fii_dii_activity(category, activity_date)')
        
        logger.info(f"EOD Extra Data tables initialized in: {self.db_path}")
    
    def _load_progress(self) -> Dict:
        """Load download progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
        return {
            'fno_bhav_copy': {'last_date': None, 'total_downloads': 0},
            'equity_bhav_copy_delivery': {'last_date': None, 'total_downloads': 0},
            'bhav_copy_indices': {'last_date': None, 'total_downloads': 0},
            'fii_dii_activity': {'last_date': None, 'total_downloads': 0}
        }
    
    def _save_progress(self):
        """Save download progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def download_fno_bhav_copy(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download FNO Bhav Copy data for date range."""
        logger.info(f"Downloading FNO Bhav Copy from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        current_dt = start_dt
        
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        while current_dt <= end_dt:
            try:
                # Skip weekends
                if current_dt.weekday() >= 5:
                    current_dt += timedelta(days=1)
                    continue
                
                trade_date_str = current_dt.strftime('%d-%m-%Y')
                logger.info(f"Downloading FNO Bhav Copy for {trade_date_str}")
                
                # Download data
                df = self.nse.fno_bhav_copy(trade_date_str)
                
                if not df.empty:
                    # Add trade date and timestamp
                    df['TRADE_DATE'] = current_dt.date()
                    df['last_updated'] = datetime.now().isoformat()
                    
                    # Store in database
                    self.conn.execute("DELETE FROM fno_bhav_copy WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO fno_bhav_copy SELECT * FROM df")
                    
                    records_added = len(df)
                    total_records += records_added
                    successful_dates += 1
                    
                    logger.info(f"✅ FNO Bhav Copy for {trade_date_str}: {records_added} records")
                    
                    # Update progress
                    self.progress['fno_bhav_copy']['last_date'] = trade_date_str
                    self.progress['fno_bhav_copy']['total_downloads'] += 1
                    self._save_progress()
                
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                failed_dates += 1
                logger.error(f"❌ Failed to download FNO Bhav Copy for {trade_date_str}: {e}")
            
            current_dt += timedelta(days=1)
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }
    
    def download_equity_bhav_copy_delivery(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download Equity Bhav Copy with Delivery data for date range."""
        logger.info(f"Downloading Equity Bhav Copy with Delivery from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        current_dt = start_dt
        
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        while current_dt <= end_dt:
            try:
                # Skip weekends
                if current_dt.weekday() >= 5:
                    current_dt += timedelta(days=1)
                    continue
                
                trade_date_str = current_dt.strftime('%d-%m-%Y')
                logger.info(f"Downloading Equity Bhav Copy with Delivery for {trade_date_str}")
                
                # Download data
                df = self.nse.bhav_copy_with_delivery(trade_date_str)
                
                if not df.empty:
                    # Add trade date and timestamp
                    df['TRADE_DATE'] = current_dt.date()
                    df['last_updated'] = datetime.now().isoformat()
                    
                    # Store in database
                    self.conn.execute("DELETE FROM equity_bhav_copy_delivery WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO equity_bhav_copy_delivery SELECT * FROM df")
                    
                    records_added = len(df)
                    total_records += records_added
                    successful_dates += 1
                    
                    logger.info(f"✅ Equity Bhav Copy with Delivery for {trade_date_str}: {records_added} records")
                    
                    # Update progress
                    self.progress['equity_bhav_copy_delivery']['last_date'] = trade_date_str
                    self.progress['equity_bhav_copy_delivery']['total_downloads'] += 1
                    self._save_progress()
                
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                failed_dates += 1
                logger.error(f"❌ Failed to download Equity Bhav Copy with Delivery for {trade_date_str}: {e}")
            
            current_dt += timedelta(days=1)
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }
    
    def download_bhav_copy_indices(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download Bhav Copy Indices data for date range."""
        logger.info(f"Downloading Bhav Copy Indices from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        current_dt = start_dt
        
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        while current_dt <= end_dt:
            try:
                # Skip weekends
                if current_dt.weekday() >= 5:
                    current_dt += timedelta(days=1)
                    continue
                
                trade_date_str = current_dt.strftime('%d-%m-%Y')
                logger.info(f"Downloading Bhav Copy Indices for {trade_date_str}")
                
                # Download data
                df = self.nse.bhav_copy_indices(trade_date_str)
                
                if not df.empty:
                    # Add trade date and timestamp
                    df['TRADE_DATE'] = current_dt.date()
                    df['last_updated'] = datetime.now().isoformat()
                    
                    # Store in database
                    self.conn.execute("DELETE FROM bhav_copy_indices WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO bhav_copy_indices SELECT * FROM df")
                    
                    records_added = len(df)
                    total_records += records_added
                    successful_dates += 1
                    
                    logger.info(f"✅ Bhav Copy Indices for {trade_date_str}: {records_added} records")
                    
                    # Update progress
                    self.progress['bhav_copy_indices']['last_date'] = trade_date_str
                    self.progress['bhav_copy_indices']['total_downloads'] += 1
                    self._save_progress()
                
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                failed_dates += 1
                logger.error(f"❌ Failed to download Bhav Copy Indices for {trade_date_str}: {e}")
            
            current_dt += timedelta(days=1)
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }
    
    def download_fii_dii_activity(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download FII DII Activity data for date range."""
        logger.info(f"Downloading FII DII Activity from {start_date} to {end_date}")
        
        # FII DII activity is typically daily data, so we'll download for each trading day
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        current_dt = start_dt
        
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        while current_dt <= end_dt:
            try:
                # Skip weekends
                if current_dt.weekday() >= 5:
                    current_dt += timedelta(days=1)
                    continue
                
                logger.info(f"Downloading FII DII Activity for {current_dt.strftime('%Y-%m-%d')}")
                
                # Download data
                df = self.nse.fii_dii_activity()
                
                if not df.empty:
                    # Add activity date and timestamp
                    df['activity_date'] = current_dt.date()
                    df['last_updated'] = datetime.now().isoformat()
                    
                    # Store in database
                    self.conn.execute("DELETE FROM fii_dii_activity WHERE activity_date = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO fii_dii_activity SELECT * FROM df")
                    
                    records_added = len(df)
                    total_records += records_added
                    successful_dates += 1
                    
                    logger.info(f"✅ FII DII Activity for {current_dt.strftime('%Y-%m-%d')}: {records_added} records")
                    
                    # Update progress
                    self.progress['fii_dii_activity']['last_date'] = current_dt.strftime('%Y-%m-%d')
                    self.progress['fii_dii_activity']['total_downloads'] += 1
                    self._save_progress()
                
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                failed_dates += 1
                logger.error(f"❌ Failed to download FII DII Activity for {current_dt.strftime('%Y-%m-%d')}: {e}")
            
            current_dt += timedelta(days=1)
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }
    
    def download_all_eod_data(self, years: int = 5) -> Dict[str, Any]:
        """Download all EOD extra data for specified number of years."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        
        logger.info(f"Starting comprehensive EOD extra data download for {years} years")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        results = {}
        
        # Download FNO Bhav Copy
        logger.info("=" * 60)
        logger.info("DOWNLOADING FNO BHAV COPY")
        logger.info("=" * 60)
        results['fno_bhav_copy'] = self.download_fno_bhav_copy(start_date, end_date)
        
        # Download Equity Bhav Copy with Delivery
        logger.info("=" * 60)
        logger.info("DOWNLOADING EQUITY BHAV COPY WITH DELIVERY")
        logger.info("=" * 60)
        results['equity_bhav_copy_delivery'] = self.download_equity_bhav_copy_delivery(start_date, end_date)
        
        # Download Bhav Copy Indices
        logger.info("=" * 60)
        logger.info("DOWNLOADING BHAV COPY INDICES")
        logger.info("=" * 60)
        results['bhav_copy_indices'] = self.download_bhav_copy_indices(start_date, end_date)
        
        # Download FII DII Activity
        logger.info("=" * 60)
        logger.info("DOWNLOADING FII DII ACTIVITY")
        logger.info("=" * 60)
        results['fii_dii_activity'] = self.download_fii_dii_activity(start_date, end_date)
        
        # Generate summary
        total_records = sum(r['total_records'] for r in results.values())
        total_successful = sum(r['successful_dates'] for r in results.values())
        total_failed = sum(r['failed_dates'] for r in results.values())
        
        summary = {
            'total_records': total_records,
            'total_successful_dates': total_successful,
            'total_failed_dates': total_failed,
            'detailed_results': results
        }
        
        logger.info("=" * 60)
        logger.info("EOD EXTRA DATA DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Records Downloaded: {total_records:,}")
        logger.info(f"Total Successful Dates: {total_successful}")
        logger.info(f"Total Failed Dates: {total_failed}")
        
        return summary
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for EOD extra data."""
        try:
            stats = {}
            
            # FNO Bhav Copy stats
            fno_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date
                FROM fno_bhav_copy
            """).fetchone()
            
            stats['fno_bhav_copy'] = {
                'total_records': fno_stats[0],
                'unique_dates': fno_stats[1],
                'earliest_date': fno_stats[2],
                'latest_date': fno_stats[3]
            }
            
            # Equity Bhav Copy with Delivery stats
            equity_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date
                FROM equity_bhav_copy_delivery
            """).fetchone()
            
            stats['equity_bhav_copy_delivery'] = {
                'total_records': equity_stats[0],
                'unique_dates': equity_stats[1],
                'earliest_date': equity_stats[2],
                'latest_date': equity_stats[3]
            }
            
            # Bhav Copy Indices stats
            indices_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date
                FROM bhav_copy_indices
            """).fetchone()
            
            stats['bhav_copy_indices'] = {
                'total_records': indices_stats[0],
                'unique_dates': indices_stats[1],
                'earliest_date': indices_stats[2],
                'latest_date': indices_stats[3]
            }
            
            # FII DII Activity stats
            fii_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT activity_date) as unique_dates,
                       MIN(activity_date) as earliest_date,
                       MAX(activity_date) as latest_date
                FROM fii_dii_activity
            """).fetchone()
            
            stats['fii_dii_activity'] = {
                'total_records': fii_stats[0],
                'unique_dates': fii_stats[1],
                'earliest_date': fii_stats[2],
                'latest_date': fii_stats[3]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def show_progress(self):
        """Show current download progress."""
        logger.info("=" * 60)
        logger.info("EOD EXTRA DATA DOWNLOAD PROGRESS")
        logger.info("=" * 60)
        
        for data_type, progress in self.progress.items():
            logger.info(f"{data_type.upper()}:")
            logger.info(f"  Last Date: {progress['last_date'] or 'None'}")
            logger.info(f"  Total Downloads: {progress['total_downloads']}")
            logger.info()


def main():
    """Main function."""
    print("EOD EXTRA DATA DOWNLOADER")
    print("=" * 60)
    
    # Initialize downloader
    downloader = EODExtraDataDownloader()
    
    # Show current progress
    downloader.show_progress()
    
    # Get database stats
    stats = downloader.get_database_stats()
    if stats:
        print(f"CURRENT DATABASE STATS:")
        for data_type, data_stats in stats.items():
            print(f"{data_type.upper()}:")
            print(f"  Total Records: {data_stats['total_records']:,}")
            print(f"  Unique Dates: {data_stats['unique_dates']}")
            if data_stats['earliest_date']:
                print(f"  Date Range: {data_stats['earliest_date']} to {data_stats['latest_date']}")
            print()
    
    # Get command line arguments for years
    years = 5  # Default to 5 years
    if len(sys.argv) > 1:
        try:
            years = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}. Using default: 5 years")
    
    print(f"Starting EOD extra data download for {years} years...")
    print(f"This will download data for all four data types.")
    print(f"Using DuckDB for faster processing...")
    
    # Start download
    results = downloader.download_all_eod_data(years)
    
    # Show final stats
    print(f"FINAL DATABASE STATS:")
    final_stats = downloader.get_database_stats()
    if final_stats:
        for data_type, data_stats in final_stats.items():
            print(f"{data_type.upper()}:")
            print(f"  Total Records: {data_stats['total_records']:,}")
            print(f"  Unique Dates: {data_stats['unique_dates']}")
            if data_stats['earliest_date']:
                print(f"  Date Range: {data_stats['earliest_date']} to {data_stats['latest_date']}")
            print()
    
    print(f"EOD EXTRA DATA DOWNLOAD COMPLETED!")


if __name__ == "__main__":
    main()
