#!/usr/bin/env python3
"""
Indian Market Data Manager
Comprehensive data management system for Indian markets.
"""

import asyncio
import aiohttp
import aiofiles
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import time
from dataclasses import dataclass
from pathlib import Path

# Import NSE utilities
try:
    from nsepy import get_history
    from nsepy.history import get_price_list
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False
    logger.warning("NSEpy not available. Install with: pip install nsepy")

try:
    from nseutils import NseUtils
    NSEUTILS_AVAILABLE = True
except ImportError:
    NSEUTILS_AVAILABLE = False
    logger.warning("NSEUtils not available. Install with: pip install nseutils")

@dataclass
class SecurityInfo:
    """Security information structure."""
    symbol: str
    name: str
    isin: str
    sector: str
    market_cap: float
    listing_date: str
    is_active: bool = True

@dataclass
class PriceData:
    """Price data structure."""
    symbol: str
    date: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    turnover: float
    last_updated: str

class IndianMarketDataManager:
    """
    Comprehensive Indian market data manager.
    Handles data collection, storage, and retrieval for all Indian securities.
    """
    
    def __init__(self, db_path: str = "data/indian_market.db"):
        """Initialize the data manager."""
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path)
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Data sources
        self.nse_utils = None
        if NSEUTILS_AVAILABLE:
            try:
                self.nse_utils = NseUtils()
                logger.info("✅ NSEUtils initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize NSEUtils: {e}")
        
        # Performance tracking
        self.stats = {
            'total_securities': 0,
            'data_points': 0,
            'last_update': None,
            'download_speed': 0
        }
    
    def _init_database(self):
        """Initialize SQLite database with optimized schema."""
        logger.info("🔧 Initializing Indian Market Database...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Securities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS securities (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    isin TEXT UNIQUE,
                    sector TEXT,
                    market_cap REAL,
                    listing_date TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    last_updated TEXT
                )
            """)
            
            # Price data table with optimized indexing
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
            
            # Create indexes for fast retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON price_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON price_data(date)")
            
            # Market metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    last_updated TEXT
                )
            """)
            
            conn.commit()
            logger.info("✅ Database initialized successfully")
    
    async def get_all_securities(self) -> List[SecurityInfo]:
        """Get all available securities from NSE."""
        logger.info("📊 Fetching all Indian securities...")
        
        securities = []
        
        try:
            if self.nse_utils:
                # Get NSE equity list
                equity_list = self.nse_utils.equity_list()
                
                for equity in equity_list:
                    try:
                        security = SecurityInfo(
                            symbol=equity.get('symbol', ''),
                            name=equity.get('companyName', ''),
                            isin=equity.get('isin', ''),
                            sector=equity.get('sector', ''),
                            market_cap=float(equity.get('marketCap', 0)),
                            listing_date=equity.get('listingDate', ''),
                            is_active=True
                        )
                        securities.append(security)
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing equity {equity}: {e}")
                        continue
                
                logger.info(f"✅ Fetched {len(securities)} securities from NSE")
            
            # Also get from NSEpy if available
            if NSEPY_AVAILABLE:
                try:
                    price_list = get_price_list(datetime.now())
                    for symbol in price_list.index:
                        if symbol not in [s.symbol for s in securities]:
                            security = SecurityInfo(
                                symbol=symbol,
                                name=symbol,
                                isin='',
                                sector='',
                                market_cap=0.0,
                                listing_date='',
                                is_active=True
                            )
                            securities.append(security)
                except Exception as e:
                    logger.warning(f"⚠️ Error fetching from NSEpy: {e}")
            
            # Store securities in database
            await self._store_securities(securities)
            
            self.stats['total_securities'] = len(securities)
            return securities
            
        except Exception as e:
            logger.error(f"❌ Error fetching securities: {e}")
            return []
    
    async def _store_securities(self, securities: List[SecurityInfo]):
        """Store securities in database."""
        with sqlite3.connect(self.db_path) as conn:
            for security in securities:
                conn.execute("""
                    INSERT OR REPLACE INTO securities 
                    (symbol, name, isin, sector, market_cap, listing_date, is_active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    security.symbol,
                    security.name,
                    security.isin,
                    security.sector,
                    security.market_cap,
                    security.listing_date,
                    security.is_active,
                    datetime.now().isoformat()
                ))
            conn.commit()
    
    async def download_historical_data(self, 
                                     symbols: List[str] = None,
                                     start_date: str = None,
                                     end_date: str = None,
                                     max_workers: int = 10) -> Dict:
        """
        Download historical data for multiple symbols efficiently.
        
        Args:
            symbols: List of symbols to download. If None, downloads all.
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_workers: Maximum concurrent downloads
        """
        logger.info("📥 Starting historical data download...")
        
        # Set default dates (10 years back)
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get symbols to download
        if not symbols:
            securities = await self.get_all_securities()
            symbols = [s.symbol for s in securities]
        
        logger.info(f"📊 Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Track progress
        total_symbols = len(symbols)
        completed = 0
        failed = 0
        start_time = time.time()
        
        # Download data concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks
            future_to_symbol = {
                executor.submit(self._download_symbol_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        completed += 1
                        logger.info(f"✅ Downloaded {symbol}: {result} records")
                    else:
                        failed += 1
                        logger.warning(f"⚠️ Failed to download {symbol}")
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error downloading {symbol}: {e}")
                
                # Progress update
                if (completed + failed) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (completed + failed) / elapsed
                    logger.info(f"📈 Progress: {completed + failed}/{total_symbols} ({rate:.2f}/sec)")
        
        # Update statistics
        elapsed = time.time() - start_time
        self.stats['download_speed'] = total_symbols / elapsed
        self.stats['last_update'] = datetime.now().isoformat()
        
        logger.info(f"🎉 Download completed!")
        logger.info(f"   ✅ Success: {completed}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   ⏱️ Time: {elapsed:.2f}s")
        logger.info(f"   🚀 Speed: {self.stats['download_speed']:.2f} symbols/sec")
        
        return {
            'total_symbols': total_symbols,
            'completed': completed,
            'failed': failed,
            'elapsed_time': elapsed,
            'download_speed': self.stats['download_speed']
        }
    
    def _download_symbol_data(self, symbol: str, start_date: str, end_date: str) -> int:
        """Download historical data for a single symbol."""
        try:
            if NSEPY_AVAILABLE:
                # Convert dates
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                # Download data
                data = get_history(symbol=symbol, start=start_dt, end=end_dt)
                
                if data.empty:
                    return 0
                
                # Store in database
                records_added = self._store_price_data(symbol, data)
                return records_added
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {e}")
            return 0
    
    def _store_price_data(self, symbol: str, data: pd.DataFrame) -> int:
        """Store price data in database efficiently."""
        records_added = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for index, row in data.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (symbol, date, open_price, high_price, low_price, close_price, volume, turnover, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        index.strftime('%Y-%m-%d'),
                        float(row.get('Open', 0)),
                        float(row.get('High', 0)),
                        float(row.get('Low', 0)),
                        float(row.get('Close', 0)),
                        int(row.get('Volume', 0)),
                        float(row.get('Turnover', 0)),
                        datetime.now().isoformat()
                    ))
                    records_added += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error storing data for {symbol} on {index}: {e}")
                    continue
            
            conn.commit()
        
        return records_added
    
    async def get_price_data(self, 
                           symbol: str, 
                           start_date: str = None, 
                           end_date: str = None) -> pd.DataFrame:
        """Get price data for a symbol from database."""
        try:
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
            logger.error(f"❌ Error getting price data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_price_data(self, 
                                    symbols: List[str],
                                    start_date: str = None,
                                    end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Get price data for multiple symbols efficiently."""
        logger.info(f"📊 Fetching price data for {len(symbols)} symbols...")
        
        results = {}
        
        # Use ThreadPoolExecutor for concurrent database queries
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.get_price_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"❌ Error getting data for {symbol}: {e}")
                    results[symbol] = pd.DataFrame()
        
        logger.info(f"✅ Retrieved data for {len(results)} symbols")
        return results
    
    async def update_daily_data(self) -> Dict:
        """Update daily data for all active securities."""
        logger.info("🔄 Updating daily data...")
        
        # Get active securities
        with sqlite3.connect(self.db_path) as conn:
            active_symbols = pd.read_sql_query(
                "SELECT symbol FROM securities WHERE is_active = 1", 
                conn
            )['symbol'].tolist()
        
        # Get last update date for each symbol
        last_dates = {}
        with sqlite3.connect(self.db_path) as conn:
            for symbol in active_symbols:
                result = conn.execute(
                    "SELECT MAX(date) FROM price_data WHERE symbol = ?", 
                    (symbol,)
                ).fetchone()
                last_dates[symbol] = result[0] if result[0] else None
        
        # Calculate missing dates
        today = datetime.now().strftime('%Y-%m-%d')
        symbols_to_update = []
        
        for symbol, last_date in last_dates.items():
            if not last_date or last_date < today:
                symbols_to_update.append(symbol)
        
        if not symbols_to_update:
            logger.info("✅ All data is up to date")
            return {'updated': 0, 'total': len(active_symbols)}
        
        logger.info(f"📥 Updating {len(symbols_to_update)} symbols...")
        
        # Download missing data
        result = await self.download_historical_data(
            symbols=symbols_to_update,
            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=today
        )
        
        return {
            'updated': result['completed'],
            'total': len(active_symbols),
            'failed': result['failed']
        }
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total securities
            total_securities = conn.execute(
                "SELECT COUNT(*) FROM securities WHERE is_active = 1"
            ).fetchone()[0]
            
            # Total data points
            total_data_points = conn.execute(
                "SELECT COUNT(*) FROM price_data"
            ).fetchone()[0]
            
            # Date range
            date_range = conn.execute(
                "SELECT MIN(date), MAX(date) FROM price_data"
            ).fetchone()
            
            # Database size
            db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
        
        return {
            'total_securities': total_securities,
            'total_data_points': total_data_points,
            'date_range': {
                'start': date_range[0],
                'end': date_range[1]
            },
            'database_size_mb': db_size,
            'last_update': self.stats['last_update'],
            'download_speed': self.stats['download_speed']
        }

# Global instance
indian_data_manager = IndianMarketDataManager() 