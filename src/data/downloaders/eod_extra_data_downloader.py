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
import threading
from queue import Queue
import math
import threading
from queue import Queue
import math

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
        self.max_workers = 10  # Increased for better performance
        self.retry_attempts = 3
        self.delay_between_requests = 0.2  # Reduced delay for faster downloads
        self.db_lock = threading.Lock()  # Thread-safe database operations
        
        # Initialize database
        self._init_database()
        
        # Load progress
        self.progress = self._load_progress()
        
    def _init_database(self):
        """Initialize DuckDB with EOD extra data tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to DuckDB
        self.conn = duckdb.connect(self.db_path)
        
        # Drop and recreate FNO Bhav Copy table to ensure correct schema
        self.conn.execute('DROP TABLE IF EXISTS fno_bhav_copy')
        self.conn.execute('''
            CREATE TABLE fno_bhav_copy (
                TradDt VARCHAR,
                BizDt VARCHAR,
                Sgmt VARCHAR,
                Src VARCHAR,
                FinInstrmTp VARCHAR,
                FinInstrmId VARCHAR,
                ISIN VARCHAR,
                TckrSymb VARCHAR,
                SctySrs VARCHAR,
                XpryDt VARCHAR,
                FininstrmActlXpryDt VARCHAR,
                StrkPric DOUBLE,
                OptnTp VARCHAR,
                FinInstrmNm VARCHAR,
                OpnPric DOUBLE,
                HghPric DOUBLE,
                LwPric DOUBLE,
                ClsPric DOUBLE,
                LastPric DOUBLE,
                PrvsClsgPric DOUBLE,
                UndrlygPric DOUBLE,
                SttlmPric DOUBLE,
                OpnIntrst BIGINT,
                ChngInOpnIntrst BIGINT,
                TtlTradgVol BIGINT,
                TtlTrfVal DOUBLE,
                TtlNbOfTxsExctd BIGINT,
                SsnId VARCHAR,
                NewBrdLotQty BIGINT,
                Rmks VARCHAR,
                Rsvd1 VARCHAR,
                Rsvd2 VARCHAR,
                Rsvd3 VARCHAR,
                Rsvd4 VARCHAR,
                TRADE_DATE DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (FinInstrmId, TRADE_DATE)
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
                PE_Ratio DOUBLE,
                PB_Ratio DOUBLE,
                Div_Yield DOUBLE,
                TRADE_DATE DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (Index_Name, TRADE_DATE)
            )
        ''')
        
        # Create FII DII Activity table
        self.conn.execute('DROP TABLE IF EXISTS fii_dii_activity')
        self.conn.execute('''
            CREATE TABLE fii_dii_activity (
                category VARCHAR,
                date VARCHAR,
                buyValue DOUBLE,
                sellValue DOUBLE,
                netValue DOUBLE,
                TRADE_DATE DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (category, date)
            )
        ''')
        
        # Create indexes for better performance
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_fno_symbol_date ON fno_bhav_copy(TckrSymb, TRADE_DATE)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_equity_symbol_date ON equity_bhav_copy_delivery(SYMBOL, TRADE_DATE)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_indices_name_date ON bhav_copy_indices(Index_Name, TRADE_DATE)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_fii_category_date ON fii_dii_activity(category, date)')
        
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

    def _download_single_date_fno(self, date_tuple: Tuple[datetime, str]) -> Dict[str, Any]:
        """Download FNO data for a single date (thread-safe)."""
        current_dt, trade_date_str = date_tuple
        result = {'success': False, 'records': 0, 'date': trade_date_str, 'error': None}
        
        try:
            # Create thread-local NSE instance
            nse = NseUtils()
            
            # Download data
            df = nse.fno_bhav_copy(trade_date_str)
            
            if not df.empty:
                # Add trade date and timestamp
                df['TRADE_DATE'] = current_dt.date()
                df['last_updated'] = datetime.now().isoformat()
                
                # Thread-safe database operations
                with self.db_lock:
                    self.conn.execute("DELETE FROM fno_bhav_copy WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO fno_bhav_copy SELECT * FROM df")
                
                result['success'] = True
                result['records'] = len(df)
                logger.info(f"SUCCESS: FNO Bhav Copy for {trade_date_str}: {len(df)} records")
            
            time.sleep(self.delay_between_requests)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"FAILED: FNO Bhav Copy for {trade_date_str}: {e}")
        
        return result

    def _download_single_date_equity(self, date_tuple: Tuple[datetime, str]) -> Dict[str, Any]:
        """Download Equity data for a single date (thread-safe)."""
        current_dt, trade_date_str = date_tuple
        result = {'success': False, 'records': 0, 'date': trade_date_str, 'error': None}
        
        try:
            # Create thread-local NSE instance
            nse = NseUtils()
            
            # Download data
            df = nse.bhav_copy_with_delivery(trade_date_str)
            
            if not df.empty:
                # Clean problematic data
                df['DELIV_QTY'] = df['DELIV_QTY'].replace(['-', ' -', '- '], '0')
                df['DELIV_QTY'] = pd.to_numeric(df['DELIV_QTY'], errors='coerce').fillna(0).astype(int)
                df['DELIV_PER'] = df['DELIV_PER'].replace(['-', ' -', '- '], '0')
                df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'], errors='coerce').fillna(0)
                
                # Add trade date and timestamp
                df['TRADE_DATE'] = current_dt.date()
                df['last_updated'] = datetime.now().isoformat()
                
                # Thread-safe database operations
                with self.db_lock:
                    self.conn.execute("DELETE FROM equity_bhav_copy_delivery WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO equity_bhav_copy_delivery SELECT * FROM df")
                
                result['success'] = True
                result['records'] = len(df)
                logger.info(f"SUCCESS: Equity Bhav Copy for {trade_date_str}: {len(df)} records")
            
            time.sleep(self.delay_between_requests)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"FAILED: Equity Bhav Copy for {trade_date_str}: {e}")
        
        return result

    def _download_single_date_indices(self, date_tuple: Tuple[datetime, str]) -> Dict[str, Any]:
        """Download Indices data for a single date (thread-safe)."""
        current_dt, trade_date_str = date_tuple
        result = {'success': False, 'records': 0, 'date': trade_date_str, 'error': None}
        
        try:
            # Create thread-local NSE instance
            nse = NseUtils()
            
            # Download data
            df = nse.bhav_copy_indices(trade_date_str)
            
            if not df.empty:
                # Clean problematic data
                numeric_columns = ['Open Index Value', 'High Index Value', 'Low Index Value', 
                                 'Closing Index Value', 'Points Change', 'Change(%)', 'Volume', 
                                 'Turnover (Rs. Cr.)', 'P/E', 'P/B', 'Div Yield']
                
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].replace(['-', ' -', '- '], '0')
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Add trade date and timestamp
                df['TRADE_DATE'] = current_dt.date()
                df['last_updated'] = datetime.now().isoformat()
                
                # Thread-safe database operations
                with self.db_lock:
                    self.conn.execute("DELETE FROM bhav_copy_indices WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO bhav_copy_indices SELECT * FROM df")
                
                result['success'] = True
                result['records'] = len(df)
                logger.info(f"SUCCESS: Bhav Copy Indices for {trade_date_str}: {len(df)} records")
            
            time.sleep(self.delay_between_requests)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"FAILED: Bhav Copy Indices for {trade_date_str}: {e}")
        
        return result

    def download_fno_bhav_copy_threaded(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download FNO Bhav Copy data using threading for better performance."""
        logger.info(f"Downloading FNO Bhav Copy (THREADED) from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate all trading dates
        trading_dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Skip weekends
                trade_date_str = current_dt.strftime('%d-%m-%Y')
                trading_dates.append((current_dt, trade_date_str))
            current_dt += timedelta(days=1)
        
        logger.info(f"Total trading days to download: {len(trading_dates)}")
        
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_date = {executor.submit(self._download_single_date_fno, date_tuple): date_tuple 
                            for date_tuple in trading_dates}
            
            # Process completed tasks
            for future in as_completed(future_to_date):
                result = future.result()
                if result['success']:
                    total_records += result['records']
                    successful_dates += 1
                    
                    # Update progress
                    self.progress['fno_bhav_copy']['last_date'] = result['date']
                    self.progress['fno_bhav_copy']['total_downloads'] += 1
                else:
                    failed_dates += 1
                
                # Save progress periodically
                if (successful_dates + failed_dates) % 10 == 0:
                    self._save_progress()
        
        self._save_progress()
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }

    def download_equity_bhav_copy_delivery_threaded(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download Equity Bhav Copy with Delivery data using threading."""
        logger.info(f"Downloading Equity Bhav Copy with Delivery (THREADED) from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate all trading dates
        trading_dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Skip weekends
                trade_date_str = current_dt.strftime('%d-%m-%Y')
                trading_dates.append((current_dt, trade_date_str))
            current_dt += timedelta(days=1)
        
        logger.info(f"Total trading days to download: {len(trading_dates)}")
        
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_date = {executor.submit(self._download_single_date_equity, date_tuple): date_tuple 
                            for date_tuple in trading_dates}
            
            # Process completed tasks
            for future in as_completed(future_to_date):
                result = future.result()
                if result['success']:
                    total_records += result['records']
                    successful_dates += 1
                    
                    # Update progress
                    self.progress['equity_bhav_copy_delivery']['last_date'] = result['date']
                    self.progress['equity_bhav_copy_delivery']['total_downloads'] += 1
                else:
                    failed_dates += 1
                
                # Save progress periodically
                if (successful_dates + failed_dates) % 10 == 0:
                    self._save_progress()
        
        self._save_progress()
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }

    def download_bhav_copy_indices_threaded(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download Bhav Copy Indices data using threading."""
        logger.info(f"Downloading Bhav Copy Indices (THREADED) from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate all trading dates
        trading_dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Skip weekends
                trade_date_str = current_dt.strftime('%d-%m-%Y')
                trading_dates.append((current_dt, trade_date_str))
            current_dt += timedelta(days=1)
        
        logger.info(f"Total trading days to download: {len(trading_dates)}")
        
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_date = {executor.submit(self._download_single_date_indices, date_tuple): date_tuple 
                            for date_tuple in trading_dates}
            
            # Process completed tasks
            for future in as_completed(future_to_date):
                result = future.result()
                if result['success']:
                    total_records += result['records']
                    successful_dates += 1
                    
                    # Update progress
                    self.progress['bhav_copy_indices']['last_date'] = result['date']
                    self.progress['bhav_copy_indices']['total_downloads'] += 1
                else:
                    failed_dates += 1
                
                # Save progress periodically
                if (successful_dates + failed_dates) % 10 == 0:
                    self._save_progress()
        
        self._save_progress()
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }

    def download_all_eod_data_threaded(self, years: int = 5) -> Dict[str, Any]:
        """Download all EOD extra data using threading for maximum performance."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        
        logger.info(f"Starting THREADED EOD extra data download for {years} years")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Using {self.max_workers} threads for parallel downloads")
        
        start_time = time.time()
        results = {}
        
        # Download FNO Bhav Copy (Threaded)
        logger.info("=" * 60)
        logger.info("DOWNLOADING FNO BHAV COPY (THREADED)")
        logger.info("=" * 60)
        results['fno_bhav_copy'] = self.download_fno_bhav_copy_threaded(start_date, end_date)
        
        # Download Equity Bhav Copy with Delivery (Threaded)
        logger.info("=" * 60)
        logger.info("DOWNLOADING EQUITY BHAV COPY WITH DELIVERY (THREADED)")
        logger.info("=" * 60)
        results['equity_bhav_copy_delivery'] = self.download_equity_bhav_copy_delivery_threaded(start_date, end_date)
        
        # Download Bhav Copy Indices (Threaded)
        logger.info("=" * 60)
        logger.info("DOWNLOADING BHAV COPY INDICES (THREADED)")
        logger.info("=" * 60)
        results['bhav_copy_indices'] = self.download_bhav_copy_indices_threaded(start_date, end_date)
        
        # Download FII DII Activity (Single download - no threading needed)
        logger.info("=" * 60)
        logger.info("DOWNLOADING FII DII ACTIVITY")
        logger.info("=" * 60)
        results['fii_dii_activity'] = self.download_fii_dii_activity(start_date, end_date)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Generate summary
        total_records = sum(r['total_records'] for r in results.values())
        total_successful = sum(r['successful_dates'] for r in results.values())
        total_failed = sum(r['failed_dates'] for r in results.values())
        
        summary = {
            'total_records': total_records,
            'total_successful_dates': total_successful,
            'total_failed_dates': total_failed,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'total_time_hours': total_time / 3600,
            'records_per_second': total_records / total_time if total_time > 0 else 0,
            'detailed_results': results
        }
        
        logger.info("=" * 60)
        logger.info("THREADED EOD EXTRA DATA DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Records Downloaded: {total_records:,}")
        logger.info(f"Total Successful Dates: {total_successful}")
        logger.info(f"Total Failed Dates: {total_failed}")
        logger.info(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"Records per Second: {total_records/total_time:.2f}")
        logger.info(f"Threads Used: {self.max_workers}")
        
        return summary

    # Keep original methods for backward compatibility
    def download_fno_bhav_copy(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download FNO Bhav Copy data for date range (original method)."""
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
                    
                    logger.info(f"SUCCESS: FNO Bhav Copy for {trade_date_str}: {records_added} records")
                    
                    # Update progress
                    self.progress['fno_bhav_copy']['last_date'] = trade_date_str
                    self.progress['fno_bhav_copy']['total_downloads'] += 1
                    self._save_progress()
                
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                failed_dates += 1
                logger.error(f"FAILED: Failed to download FNO Bhav Copy for {trade_date_str}: {e}")
            
            current_dt += timedelta(days=1)
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }
    
    def download_equity_bhav_copy_delivery(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download Equity Bhav Copy with Delivery data for date range (original method)."""
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
                    # Clean problematic data
                    df['DELIV_QTY'] = df['DELIV_QTY'].replace(['-', ' -', '- '], '0')
                    df['DELIV_QTY'] = pd.to_numeric(df['DELIV_QTY'], errors='coerce').fillna(0).astype(int)
                    df['DELIV_PER'] = df['DELIV_PER'].replace(['-', ' -', '- '], '0')
                    df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'], errors='coerce').fillna(0)
                    
                    # Add trade date and timestamp
                    df['TRADE_DATE'] = current_dt.date()
                    df['last_updated'] = datetime.now().isoformat()
                    
                    # Store in database
                    self.conn.execute("DELETE FROM equity_bhav_copy_delivery WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO equity_bhav_copy_delivery SELECT * FROM df")
                    
                    records_added = len(df)
                    total_records += records_added
                    successful_dates += 1
                    
                    logger.info(f"SUCCESS: Equity Bhav Copy with Delivery for {trade_date_str}: {records_added} records")
                    
                    # Update progress
                    self.progress['equity_bhav_copy_delivery']['last_date'] = trade_date_str
                    self.progress['equity_bhav_copy_delivery']['total_downloads'] += 1
                    self._save_progress()
                
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                failed_dates += 1
                logger.error(f"FAILED: Failed to download Equity Bhav Copy with Delivery for {trade_date_str}: {e}")
            
            current_dt += timedelta(days=1)
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }
    
    def download_bhav_copy_indices(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download Bhav Copy Indices data for date range (original method)."""
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
                    # Clean problematic data - replace '-' with 0 for numeric columns
                    numeric_columns = ['Open Index Value', 'High Index Value', 'Low Index Value', 
                                     'Closing Index Value', 'Points Change', 'Change(%)', 'Volume', 'Turnover (Rs. Cr.)', 'P/E', 'P/B', 'Div Yield']
                    
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = df[col].replace(['-', ' -', '- '], '0')
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Add trade date and timestamp
                    df['TRADE_DATE'] = current_dt.date()
                    df['last_updated'] = datetime.now().isoformat()
                    
                    # Store in database
                    self.conn.execute("DELETE FROM bhav_copy_indices WHERE TRADE_DATE = ?", [current_dt.date()])
                    self.conn.execute("INSERT INTO bhav_copy_indices SELECT * FROM df")
                    
                    records_added = len(df)
                    total_records += records_added
                    successful_dates += 1
                    
                    logger.info(f"SUCCESS: Bhav Copy Indices for {trade_date_str}: {records_added} records")
                    
                    # Update progress
                    self.progress['bhav_copy_indices']['last_date'] = trade_date_str
                    self.progress['bhav_copy_indices']['total_downloads'] += 1
                    self._save_progress()
                
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                failed_dates += 1
                logger.error(f"FAILED: Failed to download Bhav Copy Indices for {trade_date_str}: {e}")
            
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
        
        # FII DII activity is typically daily data, but NSE only provides current day data
        # So we'll download once and store it for the current date
        total_records = 0
        successful_dates = 0
        failed_dates = 0
        
        try:
            logger.info(f"Downloading FII DII Activity for current date")
            
            # Download data
            df = self.nse.fii_dii_activity()
            
            if not df.empty:
                # Add trade date and timestamp
                df['TRADE_DATE'] = datetime.now().date()
                df['last_updated'] = datetime.now().isoformat()
                
                # Store in database - clear existing data first
                self.conn.execute("DELETE FROM fii_dii_activity")
                self.conn.execute("INSERT INTO fii_dii_activity SELECT * FROM df")
                
                records_added = len(df)
                total_records = records_added
                successful_dates = 1
                
                logger.info(f"SUCCESS: FII DII Activity for current date: {records_added} records")
                
                # Update progress
                self.progress['fii_dii_activity']['last_date'] = datetime.now().strftime('%Y-%m-%d')
                self.progress['fii_dii_activity']['total_downloads'] += 1
                self._save_progress()
            
        except Exception as e:
            failed_dates = 1
            logger.error(f"FAILED: Failed to download FII DII Activity: {e}")
        
        return {
            'success': True,
            'total_records': total_records,
            'successful_dates': successful_dates,
            'failed_dates': failed_dates
        }
    
    def download_all_eod_data(self, years: int = 5) -> Dict[str, Any]:
        """Download all EOD extra data for specified number of years (original method)."""
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
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date
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
            logger.info("")


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
    
    print(f"Starting THREADED EOD extra data download for {years} years...")
    print(f"This will download data for all four data types using {downloader.max_workers} threads.")
    print(f"Using DuckDB for faster processing...")
    
    # Start threaded download
    results = downloader.download_all_eod_data_threaded(years)
    
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
    
    print(f"THREADED EOD EXTRA DATA DOWNLOAD COMPLETED!")
    print(f"Total Time: {results['total_time_minutes']:.2f} minutes")
    print(f"Records per Second: {results['records_per_second']:.2f}")


if __name__ == "__main__":
    main()
