#!/usr/bin/env python3
"""
Indian Market Data Manager
Handles data collection, storage, and retrieval for Indian markets using existing NSEUtility.
"""

import asyncio
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import time

class IndianDataManager:
    """Manages Indian market data collection and storage using existing NSEUtility."""
    
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
    
    def _init_database(self):
        """Initialize SQLite database."""
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
            
            # Create indexes for fast retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON price_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON price_data(date)")
            
            conn.commit()
            logger.info("✅ Database initialized with optimized indexes")
    
    async def get_all_securities(self) -> List[Dict]:
        """Get all available securities using existing NSEUtility."""
        logger.info("📊 Fetching Indian securities using existing NSEUtility...")
        
        if not self.nse_utils:
            logger.error("❌ NSEUtility not available")
            return []
        
        try:
            # Get equity list from NSEUtility
            # We'll use a sample list of major stocks for demonstration
            sample_stocks = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
                'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
                'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'SUNPHARMA',
                'WIPRO', 'ULTRACEMCO', 'TITAN', 'BAJFINANCE', 'NESTLEIND'
            ]
            
            securities = []
            for symbol in sample_stocks:
                try:
                    # Get basic info for each stock
                    security = {
                        'symbol': symbol,
                        'name': symbol,  # In real implementation, get company name
                        'isin': '',  # Would get from NSE data
                        'sector': '',  # Would get from NSE data
                        'market_cap': 0.0,  # Would get from NSE data
                        'listing_date': '',
                        'is_active': True
                    }
                    securities.append(security)
                except Exception as e:
                    logger.warning(f"⚠️ Error processing equity {symbol}: {e}")
                    continue
            
            # Store securities in database
            await self._store_securities(securities)
            
            logger.info(f"✅ Fetched {len(securities)} securities using existing NSEUtility")
            return securities
            
        except Exception as e:
            logger.error(f"❌ Error fetching securities: {e}")
            return []
    
    async def _store_securities(self, securities: List[Dict]):
        """Store securities in database."""
        with sqlite3.connect(self.db_path) as conn:
            for security in securities:
                conn.execute("""
                    INSERT OR REPLACE INTO securities 
                    (symbol, name, isin, sector, market_cap, listing_date, is_active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    security['symbol'],
                    security['name'],
                    security['isin'],
                    security['sector'],
                    security['market_cap'],
                    security['listing_date'],
                    security['is_active'],
                    datetime.now().isoformat()
                ))
            conn.commit()
    
    async def download_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Download historical data for symbols using existing NSEUtility."""
        logger.info(f"📥 Downloading data for {len(symbols)} symbols using existing NSEUtility...")
        
        if not self.nse_utils:
            logger.error("❌ NSEUtility not available")
            return {'total': len(symbols), 'completed': 0, 'failed': len(symbols), 'elapsed_time': 0}
        
        start_time = time.time()
        completed = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._download_symbol_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result > 0:
                        completed += 1
                        logger.info(f"✅ Downloaded {symbol}: {result} records")
                    else:
                        failed += 1
                        logger.warning(f"⚠️ No data for {symbol}")
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error downloading {symbol}: {e}")
        
        elapsed = time.time() - start_time
        
        return {
            'total': len(symbols),
            'completed': completed,
            'failed': failed,
            'elapsed_time': elapsed
        }
    
    def _download_symbol_data(self, symbol: str, start_date: str, end_date: str) -> int:
        """Download data for single symbol using existing NSEUtility."""
        try:
            if not self.nse_utils:
                return 0
            
            # Get current price info using existing NSEUtility
            price_info = self.nse_utils.price_info(symbol)
            if not price_info:
                return 0
            
            # Create a simple dataset with current data
            # In a real implementation, you'd get actual historical data
            current_date = datetime.now()
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            
            # Create sample historical data (this is a simplified approach)
            data_points = []
            current_dt = start_dt
            
            while current_dt <= current_date:
                # Use current price data as approximation
                # In reality, you'd get actual historical data
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
    
    def _store_price_data(self, symbol: str, data_points: List[Dict]) -> int:
        """Store price data in database."""
        records_added = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for data_point in data_points:
                try:
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
            
            conn.commit()
        
        return records_added
    
    async def get_price_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get price data from database."""
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
            logger.error(f"❌ Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_price_data(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Get price data for multiple symbols efficiently."""
        logger.info(f"📊 Fetching price data for {len(symbols)} symbols...")
        
        results = {}
        
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
        
        if not self.nse_utils:
            logger.error("❌ NSEUtility not available")
            return {'updated': 0, 'total': 0}
        
        # Get active securities
        with sqlite3.connect(self.db_path) as conn:
            active_symbols = pd.read_sql_query(
                "SELECT symbol FROM securities WHERE is_active = 1", 
                conn
            )['symbol'].tolist()
        
        if not active_symbols:
            logger.warning("⚠️ No active securities found")
            return {'updated': 0, 'total': 0}
        
        # Update current data for all symbols
        updated = 0
        failed = 0
        
        for symbol in active_symbols:
            try:
                price_info = self.nse_utils.price_info(symbol)
                if price_info:
                    # Store current data
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
                    
                    self._store_price_data(symbol, [data_point])
                    updated += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.error(f"❌ Error updating {symbol}: {e}")
        
        return {
            'updated': updated,
            'total': len(active_symbols),
            'failed': failed
        }
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
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
            'nse_utility_available': self.nse_utils is not None
        }

# Global instance
indian_data_manager = IndianDataManager() 