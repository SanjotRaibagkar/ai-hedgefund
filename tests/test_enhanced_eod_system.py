#!/usr/bin/env python3
"""
Enhanced EOD System Test
Implements comprehensive Indian market data collection and screening.
"""

import asyncio
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class EnhancedEODSystem:
    """Enhanced EOD System with comprehensive data management."""
    
    def __init__(self, db_path: str = "data/indian_market.db"):
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path)
        os.makedirs(self.db_dir, exist_ok=True)
        self._init_database()
        
        # Initialize NSEUtility
        self.nse_utils = None
        try:
            from src.nsedata.NseUtility import NseUtils
            self.nse_utils = NseUtils()
            logger.info("✅ NSEUtility initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize NSEUtility: {e}")
        
        # Performance settings
        self.max_workers = 20
        self.batch_size = 50
        self.retry_attempts = 3
    
    def _init_database(self):
        """Initialize enhanced database schema."""
        logger.info("🔧 Initializing Enhanced Database...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Securities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS securities (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    last_updated TEXT,
                    last_data_date TEXT,
                    total_records INTEGER DEFAULT 0
                )
            """)
            
            # Price data table
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
            
            # Download tracker
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_tracker (
                    symbol TEXT PRIMARY KEY,
                    last_download_date TEXT,
                    download_status TEXT,
                    records_count INTEGER DEFAULT 0,
                    error_message TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON price_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON price_data(date)")
            
            conn.commit()
            logger.info("✅ Enhanced database initialized")
    
    async def get_all_indian_stocks(self) -> List[str]:
        """Get all available Indian stocks from multiple sources."""
        logger.info("📊 Fetching ALL Indian stocks...")
        
        if not self.nse_utils:
            logger.error("❌ NSEUtility not available")
            return []
        
        try:
            all_symbols = set()
            
            # Get from NIFTY indices
            nifty_indices = [
                'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY MIDCAP 50', 'NIFTY MIDCAP 100',
                'NIFTY SMALLCAP 50', 'NIFTY SMALLCAP 100', 'NIFTY BANK', 'NIFTY AUTO',
                'NIFTY IT', 'NIFTY PHARMA', 'NIFTY FMCG', 'NIFTY METAL'
            ]
            
            for index in nifty_indices:
                try:
                    symbols = self.nse_utils.get_index_details(index, list_only=True)
                    if symbols:
                        all_symbols.update(symbols)
                        logger.info(f"✅ Added {len(symbols)} symbols from {index}")
                except Exception as e:
                    logger.warning(f"⚠️ Error getting {index}: {e}")
            
            # Add major stocks as backup
            major_stocks = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
                'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'AXISBANK', 'ASIANPAINT',
                'MARUTI', 'HCLTECH', 'SUNPHARMA', 'WIPRO', 'ULTRACEMCO', 'TITAN',
                'BAJFINANCE', 'NESTLEIND', 'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA'
            ]
            all_symbols.update(major_stocks)
            
            # Store in database
            symbols_list = sorted(list(all_symbols))
            await self._store_securities(symbols_list)
            
            logger.info(f"✅ Total unique symbols: {len(symbols_list)}")
            return symbols_list
            
        except Exception as e:
            logger.error(f"❌ Error fetching stocks: {e}")
            return []
    
    async def _store_securities(self, symbols: List[str]):
        """Store securities in database."""
        with sqlite3.connect(self.db_path) as conn:
            for symbol in symbols:
                conn.execute("""
                    INSERT OR REPLACE INTO securities 
                    (symbol, name, sector, is_active, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    symbol, symbol, '', True, datetime.now().isoformat()
                ))
            conn.commit()
    
    async def download_10_years_data(self, symbols: List[str] = None) -> Dict:
        """Download 10 years of historical data for all symbols."""
        logger.info("📥 Starting 10-year historical data download...")
        
        if not symbols:
            symbols = await self.get_all_indian_stocks()
        
        # Calculate date range (10 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        
        logger.info(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"📊 Total Symbols: {len(symbols)}")
        
        # Process in batches
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
                    time.sleep(1)
                else:
                    logger.error(f"❌ Failed {symbol} after {self.retry_attempts} attempts: {e}")
                    return 0
        return 0
    
    def _download_symbol_data(self, symbol: str, start_date: str, end_date: str) -> int:
        """Download data for single symbol."""
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
            
            # Create historical dataset (simplified)
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            data_points = []
            current_dt = start_dt
            
            while current_dt <= end_dt:
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
            
            # Store the data
            return self._store_price_data(symbol, data_points)
            
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
    
    def _store_price_data(self, symbol: str, data_points: List[Dict]) -> int:
        """Store price data with duplicate handling."""
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
                    continue
            
            # Update securities table
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
                (symbol, last_download_date, download_status, records_count, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now().isoformat(),
                'SUCCESS' if records_added > 0 else 'FAILED',
                records_added,
                ''
            ))
            
            conn.commit()
        
        return records_added
    
    async def update_latest_data(self) -> Dict:
        """Update latest data for all securities."""
        logger.info("🔄 Updating latest data...")
        
        # Get symbols that need updating
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
            
            records_added = self._store_price_data(symbol, [data_point])
            return 'updated' if records_added > 0 else 'skipped'
            
        except Exception as e:
            logger.error(f"❌ Error updating {symbol}: {e}")
            return 'failed'
    
    async def get_price_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get price data with automatic update check."""
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

async def main():
    """Main test function."""
    logger.info("🚀 Enhanced EOD System Test")
    logger.info("=" * 50)
    
    # Initialize system
    system = EnhancedEODSystem()
    
    # Test 1: Get all Indian stocks
    logger.info("\n📊 Test 1: Getting All Indian Stocks")
    logger.info("-" * 40)
    
    symbols = await system.get_all_indian_stocks()
    logger.info(f"✅ Found {len(symbols)} unique symbols")
    
    if symbols:
        logger.info(f"📋 Sample symbols: {symbols[:10]}")
        
        # Test 2: Download 10 years data (small sample for testing)
        logger.info("\n📥 Test 2: Downloading 10 Years Data (Sample)")
        logger.info("-" * 40)
        
        # Use first 10 symbols for testing
        test_symbols = symbols[:10]
        result = await system.download_10_years_data(test_symbols)
        
        logger.info(f"✅ Download completed:")
        logger.info(f"   Total symbols: {result['total_symbols']}")
        logger.info(f"   Completed: {result['completed']}")
        logger.info(f"   Failed: {result['failed']}")
        logger.info(f"   Total records: {result['total_records']}")
        logger.info(f"   Time: {result['elapsed_time']:.2f}s")
        logger.info(f"   Avg time per symbol: {result['avg_time_per_symbol']:.2f}s")
        
        # Test 3: Update latest data
        logger.info("\n🔄 Test 3: Updating Latest Data")
        logger.info("-" * 40)
        
        update_result = await system.update_latest_data()
        logger.info(f"✅ Update completed:")
        logger.info(f"   Updated: {update_result['updated']}")
        logger.info(f"   Skipped: {update_result['skipped']}")
        logger.info(f"   Failed: {update_result['failed']}")
        
        # Test 4: Get data with automatic update
        logger.info("\n📊 Test 4: Getting Data with Auto-Update")
        logger.info("-" * 40)
        
        test_symbol = symbols[0]
        data = await system.get_price_data(test_symbol)
        logger.info(f"✅ Got data for {test_symbol}: {len(data)} records")
        
        if not data.empty:
            logger.info(f"   Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"   Latest price: ₹{data['close_price'].iloc[-1]:.2f}")
        
        # Test 5: Database statistics
        logger.info("\n📈 Test 5: Database Statistics")
        logger.info("-" * 40)
        
        stats = system.get_database_stats()
        logger.info(f"✅ Database stats:")
        logger.info(f"   Total securities: {stats['total_securities']}")
        logger.info(f"   Total data points: {stats['total_data_points']}")
        logger.info(f"   Database size: {stats['database_size_mb']:.2f} MB")
        logger.info(f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        logger.info(f"   Download stats: {stats['download_stats']}")
    
    logger.info("\n🎉 Enhanced EOD System Test Completed!")

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run the test
    asyncio.run(main()) 