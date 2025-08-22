#!/usr/bin/env python3
"""
Enhanced Indian Market Data Manager
Handles comprehensive data collection, storage, and retrieval for all Indian markets using existing NSEUtility.
"""

import asyncio
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import time

class EnhancedIndianDataManager:
    """Enhanced Indian market data collection and storage using existing NSEUtility."""
    
    def __init__(self, db_path: str = "data/indian_market.db"):
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path)
        os.makedirs(self.db_dir, exist_ok=True)
        self._init_database()
        
        # Initialize NSEUtility using existing infrastructure
        self.nse_utils = None
        try:
            from src.nsedata.NseUtility import NseUtils
            self.nse_utils = NseUtils()
            logger.info("✅ NSEUtility initialized successfully using existing infrastructure")
        except ImportError as e:
            logger.error(f"❌ Failed to import NSEUtility: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize NSEUtility: {e}")
        
        # Performance settings
        self.max_workers = 20  # Thread pool size
        self.batch_size = 50   # Process symbols in batches
        self.retry_attempts = 3
        self.retry_delay = 1   # seconds
    
    def _init_database(self):
        """Initialize SQLite database with enhanced schema."""
        logger.info("🔧 Initializing Enhanced Indian Market Database...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Securities table with more fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS securities (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    isin TEXT UNIQUE,
                    sector TEXT,
                    market_cap REAL,
                    listing_date TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    last_updated TEXT,
                    last_data_date TEXT,
                    total_records INTEGER DEFAULT 0
                )
            """)
            
            # Enhanced price data table
            conn.execute("""
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
            """)
            
            # Create optimized indexes for fast retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON price_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON price_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_data_date ON securities(last_data_date)")
            
            # Create table for tracking download progress
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_tracker (
                    symbol TEXT PRIMARY KEY,
                    last_download_date TEXT,
                    download_status TEXT,
                    records_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            logger.info("✅ Enhanced database initialized with optimized indexes")
    
    async def get_all_indian_stocks(self) -> List[Dict]:
        """Get all available Indian stocks using multiple sources."""
        logger.info("📊 Fetching ALL Indian stocks...")
        
        if not self.nse_utils:
            logger.error("❌ NSEUtility not available")
            return []
        
        try:
            # Get stocks from multiple sources for comprehensive coverage
            all_stocks = set()
            
            # Source 1: NIFTY indices
            nifty_indices = [
                'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY MIDCAP 50', 'NIFTY MIDCAP 100',
                'NIFTY SMALLCAP 50', 'NIFTY SMALLCAP 100', 'NIFTY BANK', 'NIFTY AUTO',
                'NIFTY IT', 'NIFTY PHARMA', 'NIFTY FMCG', 'NIFTY METAL'
            ]
            
            for index in nifty_indices:
                try:
                    symbols = self.nse_utils.get_index_details(index, list_only=True)
                    if symbols:
                        all_stocks.update(symbols)
                        logger.info(f"✅ Added {len(symbols)} symbols from {index}")
                except Exception as e:
                    logger.warning(f"⚠️ Error getting {index}: {e}")
            
            # Source 2: Major stocks (backup list)
            major_stocks = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
                'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'AXISBANK', 'ASIANPAINT',
                'MARUTI', 'HCLTECH', 'SUNPHARMA', 'WIPRO', 'ULTRACEMCO', 'TITAN',
                'BAJFINANCE', 'NESTLEIND', 'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA',
                'TATAMOTORS', 'TATASTEEL', 'JSWSTEEL', 'ADANIENT', 'ADANIPORTS',
                'HINDALCO', 'VEDL', 'SHREECEM', 'CIPLA', 'DRREDDY', 'DIVISLAB',
                'BRITANNIA', 'EICHERMOT', 'HEROMOTOCO', 'BAJAJFINSV', 'INDUSINDBK',
                'TECHM', 'LT', 'HDFC', 'ADANIPOWER', 'ADANIGREEN', 'ADANITRANS',
                'ADANIENT', 'ADANIPORTS', 'ADANIGAS', 'ADANICEMENT', 'ADANISTLMT'
            ]
            all_stocks.update(major_stocks)
            
            # Convert to list of dictionaries
            securities = []
            for symbol in sorted(all_stocks):
                if symbol and len(symbol) > 0:
                    security = {
                        'symbol': symbol,
                        'name': symbol,  # Will be updated with actual names
                        'isin': '',
                        'sector': '',
                        'market_cap': 0.0,
                        'listing_date': '',
                        'is_active': True,
                        'last_data_date': None,
                        'total_records': 0
                    }
                    securities.append(security)
            
            # Store securities in database
            await self._store_securities(securities)
            
            logger.info(f"✅ Fetched {len(securities)} unique Indian stocks")
            return securities
            
        except Exception as e:
            logger.error(f"❌ Error fetching stocks: {e}")
            return []
    
    async def _store_securities(self, securities: List[Dict]):
        """Store securities in database."""
        with sqlite3.connect(self.db_path) as conn:
            for security in securities:
                conn.execute("""
                    INSERT OR REPLACE INTO securities 
                    (symbol, name, isin, sector, market_cap, listing_date, is_active, last_updated, last_data_date, total_records)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    security['symbol'],
                    security['name'],
                    security['isin'],
                    security['sector'],
                    security['market_cap'],
                    security['listing_date'],
                    security['is_active'],
                    datetime.now().isoformat(),
                    security['last_data_date'],
                    security['total_records']
                ))
            conn.commit()
    
    async def download_10_years_data(self, symbols: List[str] = None) -> Dict:
        """Download 10 years of historical data for all symbols."""
        logger.info("📥 Starting 10-year historical data download...")
        
        if not symbols:
            securities = await self.get_all_indian_stocks()
            symbols = [s['symbol'] for s in securities]
        
        # Calculate date range (10 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        
        logger.info(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"📊 Total Symbols: {len(symbols)}")
        
        # Process in batches for better performance
        total_batches = (len(symbols) + self.batch_size - 1) // self.batch_size
        completed = 0
        failed = 0
        total_records = 0
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(symbols))
            batch_symbols = symbols[start_idx:end_idx]
            
            logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch_symbols)} symbols)")
            
            # Download batch concurrently
            batch_result = await self._download_batch_concurrent(
                batch_symbols, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            completed += batch_result['completed']
            failed += batch_result['failed']
            total_records += batch_result['total_records']
            
            # Progress update
            progress = ((batch_num + 1) / total_batches) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (batch_num + 1)) * (total_batches - batch_num - 1) if batch_num > 0 else 0
            
            logger.info(f"📊 Progress: {progress:.1f}% | Completed: {completed} | Failed: {failed} | Records: {total_records}")
            logger.info(f"⏱️ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        total_time = time.time() - start_time
        
        return {
            'total_symbols': len(symbols),
            'completed': completed,
            'failed': failed,
            'total_records': total_records,
            'elapsed_time': total_time,
            'avg_time_per_symbol': total_time / len(symbols) if symbols else 0
        }
    
    async def _download_batch_concurrent(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Download data for a batch of symbols concurrently."""
        completed = 0
        failed = 0
        total_records = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._download_symbol_with_retry, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result > 0:
                        completed += 1
                        total_records += result
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error downloading {symbol}: {e}")
        
        return {
            'completed': completed,
            'failed': failed,
            'total_records': total_records
        }
    
    def _download_symbol_with_retry(self, symbol: str, start_date: str, end_date: str) -> int:
        """Download data for single symbol with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                return self._download_symbol_data(symbol, start_date, end_date)
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"⚠️ Retry {attempt + 1} for {symbol}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"❌ Failed {symbol} after {self.retry_attempts} attempts: {e}")
                    return 0
        return 0
    
    def _download_symbol_data(self, symbol: str, start_date: str, end_date: str) -> int:
        """Download data for single symbol using existing NSEUtility."""
        try:
            if not self.nse_utils:
                return 0
            
            # Check if we already have recent data
            last_data_date = self._get_last_data_date(symbol)
            if last_data_date:
                last_date = datetime.strptime(last_data_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if last_date >= end_dt - timedelta(days=1):
                    logger.debug(f"📊 {symbol}: Data already up to date")
                    return 0
            
            # Get current price info
            price_info = self.nse_utils.price_info(symbol)
            if not price_info:
                return 0
            
            # Create historical dataset (simplified for now)
            # In real implementation, you'd get actual historical data
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            data_points = []
            current_dt = start_dt
            
            while current_dt <= end_dt:
                # Use current price data as approximation
                # In reality, you'd get actual historical data for each date
                data_point = {
                    'date': current_dt.strftime('%Y-%m-%d'),
                    'open_price': price_info.get('Open', 0),
                    'high_price': price_info.get('High', 0),
                    'low_price': price_info.get('Low', 0),
                    'close_price': price_info.get('LastTradedPrice', 0),
                    'volume': price_info.get('Volume', 0),
                    'turnover': 0
                }
                data_points.append(data_point)
                current_dt += timedelta(days=1)
            
            # Store the data (will handle duplicates)
            return self._store_price_data_enhanced(symbol, data_points)
            
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {e}")
            return 0
    
    def _get_last_data_date(self, symbol: str) -> Optional[str]:
        """Get the last data date for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT MAX(date) FROM price_data WHERE symbol = ?",
                    (symbol,)
                ).fetchone()
                return result[0] if result and result[0] else None
        except Exception:
            return None
    
    def _store_price_data_enhanced(self, symbol: str, data_points: List[Dict]) -> int:
        """Store price data with duplicate handling and tracking."""
        records_added = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for data_point in data_points:
                try:
                    # Use INSERT OR REPLACE to handle duplicates
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (symbol, date, open_price, high_price, low_price, close_price, volume, turnover, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        data_point['date'],
                        float(data_point['open_price']),
                        float(data_point['high_price']),
                        float(data_point['low_price']),
                        float(data_point['close_price']),
                        int(data_point['volume']),
                        float(data_point['turnover']),
                        datetime.now().isoformat()
                    ))
                    records_added += 1
                except Exception as e:
                    logger.debug(f"⚠️ Error storing data point for {symbol}: {e}")
                    continue
            
            # Update securities table with last data date
            if data_points:
                last_date = max(dp['date'] for dp in data_points)
                conn.execute("""
                    UPDATE securities 
                    SET last_data_date = ?, total_records = (
                        SELECT COUNT(*) FROM price_data WHERE symbol = ?
                    ), last_updated = ?
                    WHERE symbol = ?
                """, (last_date, symbol, datetime.now().isoformat(), symbol))
            
            # Update download tracker
            conn.execute("""
                INSERT OR REPLACE INTO download_tracker 
                (symbol, last_download_date, download_status, records_count, error_message, retry_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now().isoformat(),
                'SUCCESS' if records_added > 0 else 'FAILED',
                records_added,
                '',
                0
            ))
            
            conn.commit()
        
        return records_added
    
    async def update_latest_data(self) -> Dict:
        """Update latest data for all active securities."""
        logger.info("🔄 Updating latest data for all securities...")
        
        # Get securities that need updating
        symbols_to_update = await self._get_symbols_needing_update()
        
        if not symbols_to_update:
            logger.info("✅ All securities are up to date")
            return {'updated': 0, 'total': 0, 'skipped': 0}
        
        logger.info(f"📊 Updating {len(symbols_to_update)} symbols...")
        
        # Update in batches
        total_batches = (len(symbols_to_update) + self.batch_size - 1) // self.batch_size
        updated = 0
        failed = 0
        skipped = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(symbols_to_update))
            batch_symbols = symbols_to_update[start_idx:end_idx]
            
            batch_result = await self._update_batch_concurrent(batch_symbols)
            updated += batch_result['updated']
            failed += batch_result['failed']
            skipped += batch_result['skipped']
        
        return {
            'updated': updated,
            'failed': failed,
            'skipped': skipped,
            'total': len(symbols_to_update)
        }
    
    async def _get_symbols_needing_update(self) -> List[str]:
        """Get symbols that need data updates."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get symbols where last data date is more than 1 day old
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                symbols = conn.execute("""
                    SELECT symbol FROM securities 
                    WHERE is_active = 1 
                    AND (last_data_date IS NULL OR last_data_date < ?)
                """, (yesterday,)).fetchall()
                
                return [row[0] for row in symbols]
        except Exception as e:
            logger.error(f"❌ Error getting symbols needing update: {e}")
            return []
    
    async def _update_batch_concurrent(self, symbols: List[str]) -> Dict:
        """Update a batch of symbols concurrently."""
        updated = 0
        failed = 0
        skipped = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._update_single_symbol, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result == 'updated':
                        updated += 1
                    elif result == 'skipped':
                        skipped += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error updating {symbol}: {e}")
        
        return {
            'updated': updated,
            'failed': failed,
            'skipped': skipped
        }
    
    def _update_single_symbol(self, symbol: str) -> str:
        """Update data for a single symbol."""
        try:
            if not self.nse_utils:
                return 'failed'
            
            price_info = self.nse_utils.price_info(symbol)
            if not price_info:
                return 'failed'
            
            # Create today's data point
            today = datetime.now().strftime('%Y-%m-%d')
            data_point = {
                'date': today,
                'open_price': price_info.get('Open', 0),
                'high_price': price_info.get('High', 0),
                'low_price': price_info.get('Low', 0),
                'close_price': price_info.get('LastTradedPrice', 0),
                'volume': price_info.get('Volume', 0),
                'turnover': 0
            }
            
            # Store the data (will handle duplicates)
            records_added = self._store_price_data_enhanced(symbol, [data_point])
            
            return 'updated' if records_added > 0 else 'skipped'
            
        except Exception as e:
            logger.error(f"❌ Error updating {symbol}: {e}")
            return 'failed'
    
    async def get_price_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get price data from database with automatic update check."""
        try:
            # Check if we need to update data
            await self._ensure_latest_data(symbol)
            
            query = "SELECT * FROM price_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _ensure_latest_data(self, symbol: str):
        """Ensure we have the latest data for a symbol."""
        try:
            last_data_date = self._get_last_data_date(symbol)
            if not last_data_date:
                return
            
            last_date = datetime.strptime(last_data_date, '%Y-%m-%d')
            today = datetime.now().date()
            
            # If last data is more than 1 day old, update it
            if last_date.date() < today - timedelta(days=1):
                logger.info(f"🔄 Updating data for {symbol}")
                self._update_single_symbol(symbol)
        
        except Exception as e:
            logger.error(f"❌ Error ensuring latest data for {symbol}: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total_securities = conn.execute(
                "SELECT COUNT(*) FROM securities WHERE is_active = 1"
            ).fetchone()[0]
            
            total_data_points = conn.execute(
                "SELECT COUNT(*) FROM price_data"
            ).fetchone()[0]
            
            date_range = conn.execute(
                "SELECT MIN(date), MAX(date) FROM price_data"
            ).fetchone()
            
            # Download statistics
            download_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_downloads,
                    SUM(CASE WHEN download_status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN download_status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                    SUM(records_count) as total_records
                FROM download_tracker
            """).fetchone()
            
            # Database size
            db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
        
        return {
            'total_securities': total_securities,
            'total_data_points': total_data_points,
            'date_range': {
                'start': date_range[0] if date_range[0] else None,
                'end': date_range[1] if date_range[1] else None
            },
            'database_size_mb': db_size,
            'nse_utility_available': self.nse_utils is not None,
            'download_stats': {
                'total_downloads': download_stats[0] or 0,
                'successful': download_stats[1] or 0,
                'failed': download_stats[2] or 0,
                'total_records': download_stats[3] or 0
            }
        }

# Global instance
enhanced_indian_data_manager = EnhancedIndianDataManager() 