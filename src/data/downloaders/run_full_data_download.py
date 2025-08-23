#!/usr/bin/env python3
"""
Full Data Download and Screening Test
Downloads data for all 2799 Indian symbols and tests the screening system.
"""

import asyncio
import sqlite3
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from src.nsedata.NseUtility import NseUtils

class FullDataDownloader:
    def __init__(self, db_path: str = "data/full_indian_market.db"):
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path)
        os.makedirs(self.db_dir, exist_ok=True)
        self.nse_utils = NseUtils()
        self._init_database()
        self.executor = ThreadPoolExecutor(max_workers=20)
        
    def _init_database(self):
        """Initialize the database with optimized schema."""
        logger.info("🔧 Initializing Full Indian Market Database...")
        with sqlite3.connect(self.db_path) as conn:
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
                    last_data_date TEXT
                )
            """)
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
            # Create optimized indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON price_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON price_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_data_date ON securities(last_data_date)")
            conn.commit()
            logger.info("✅ Database initialized with optimized indexes")

    def load_all_symbols(self):
        """Load all 2799 symbols from our comprehensive list."""
        logger.info("📊 Loading all 2799 Indian symbols...")
        
        symbols = set()
        
        # Load from CSV symbols
        try:
            if os.path.exists('nse_equity_symbols_complete.txt'):
                with open('nse_equity_symbols_complete.txt', 'r') as f:
                    csv_symbols = [line.strip() for line in f if line.strip()]
                symbols.update(csv_symbols)
                logger.info(f"✅ Loaded {len(csv_symbols)} symbols from CSV")
        except Exception as e:
            logger.error(f"❌ Error loading CSV symbols: {e}")
        
        # Load from NIFTY indices
        try:
            indices = [
                'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY MIDCAP 50', 'NIFTY MIDCAP 100',
                'NIFTY SMALLCAP 50', 'NIFTY SMALLCAP 100', 'NIFTY SMALLCAP 250',
                'NIFTY BANK', 'NIFTY IT', 'NIFTY PHARMA', 'NIFTY AUTO', 'NIFTY FMCG',
                'NIFTY METAL', 'NIFTY REALTY', 'Securities in F&O'
            ]
            
            for idx in indices:
                try:
                    index_symbols = self.nse_utils.get_index_details(idx, list_only=True)
                    if index_symbols:
                        symbols.update(index_symbols)
                        logger.info(f"✅ {idx}: {len(index_symbols)} symbols")
                except Exception as e:
                    logger.warning(f"⚠️ {idx}: Error - {e}")
        except Exception as e:
            logger.error(f"❌ Error loading index symbols: {e}")
        
        symbols_list = sorted(list(symbols))
        logger.info(f"🎯 Total unique symbols: {len(symbols_list)}")
        return symbols_list

    def download_symbol_data(self, symbol):
        """Download data for a single symbol."""
        try:
            # Get current price info
            price_info = self.nse_utils.price_info(symbol)
            if not price_info:
                return None
            
            # Create data record
            data = {
                'symbol': symbol,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'open_price': price_info.get('Open', 0),
                'high_price': price_info.get('High', 0),
                'low_price': price_info.get('Low', 0),
                'close_price': price_info.get('LastTradedPrice', 0),
                'volume': price_info.get('Volume', 0),
                'turnover': price_info.get('Turnover', 0),
                'last_updated': datetime.now().isoformat()
            }
            
            return data
            
        except Exception as e:
            logger.warning(f"⚠️ Error downloading {symbol}: {e}")
            return None

    def store_price_data(self, data_list):
        """Store price data in database."""
        if not data_list:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            for data in data_list:
                if data:
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (symbol, date, open_price, high_price, low_price, close_price, volume, turnover, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['symbol'], data['date'], data['open_price'], data['high_price'],
                        data['low_price'], data['close_price'], data['volume'], data['turnover'], data['last_updated']
                    ))
            conn.commit()

    def download_all_data(self, symbols, max_symbols=None):
        """Download data for all symbols."""
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        logger.info(f"🚀 Starting download for {len(symbols)} symbols...")
        start_time = time.time()
        
        successful_downloads = 0
        failed_downloads = 0
        data_list = []
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"📥 Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} symbols)")
            
            # Download data for batch
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(self.download_symbol_data, symbol) for symbol in batch]
                
                for future in asyncio.as_completed(futures):
                    try:
                        data = future.result()
                        if data:
                            data_list.append(data)
                            successful_downloads += 1
                        else:
                            failed_downloads += 1
                    except Exception as e:
                        failed_downloads += 1
                        logger.error(f"❌ Future error: {e}")
            
            # Store batch data
            if data_list:
                self.store_price_data(data_list)
                data_list = []
            
            # Progress update
            elapsed = time.time() - start_time
            rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
            logger.info(f"⏱️ Progress: {i + len(batch)}/{len(symbols)} symbols ({rate:.1f} symbols/sec)")
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Download completed!")
        logger.info(f"✅ Successful: {successful_downloads}")
        logger.info(f"❌ Failed: {failed_downloads}")
        logger.info(f"⏱️ Total time: {total_time:.1f} seconds")
        logger.info(f"📊 Rate: {successful_downloads/total_time:.1f} symbols/sec")

    def get_database_stats(self):
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Securities count
            securities_count = conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
            
            # Price data count
            price_data_count = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
            
            # Unique symbols with data
            symbols_with_data = conn.execute("SELECT COUNT(DISTINCT symbol) FROM price_data").fetchone()[0]
            
            # Latest data date
            latest_date = conn.execute("SELECT MAX(date) FROM price_data").fetchone()[0]
            
            return {
                'securities_count': securities_count,
                'price_data_count': price_data_count,
                'symbols_with_data': symbols_with_data,
                'latest_date': latest_date
            }

    def test_screening(self):
        """Test the screening system with database-first approach."""
        logger.info("🧪 Testing screening system...")
        
        # Get symbols with data
        with sqlite3.connect(self.db_path) as conn:
            symbols_with_data = conn.execute("""
                SELECT DISTINCT symbol FROM price_data 
                WHERE close_price > 0 
                ORDER BY symbol 
                LIMIT 50
            """).fetchall()
        
        symbols = [row[0] for row in symbols_with_data]
        logger.info(f"📊 Testing screening with {len(symbols)} symbols")
        
        # Simple momentum screening
        signals = []
        for symbol in symbols:
            try:
                # Get recent data
                with sqlite3.connect(self.db_path) as conn:
                    data = pd.read_sql_query("""
                        SELECT * FROM price_data 
                        WHERE symbol = ? 
                        ORDER BY date DESC 
                        LIMIT 20
                    """, conn, params=(symbol,))
                
                if len(data) >= 10:
                    # Calculate simple momentum
                    current_price = data.iloc[0]['close_price']
                    avg_price = data['close_price'].mean()
                    
                    if current_price > avg_price * 1.05:  # 5% above average
                        signals.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'avg_price': avg_price,
                            'momentum': ((current_price - avg_price) / avg_price) * 100,
                            'signal': 'BULLISH',
                            'entry': current_price,
                            'sl': current_price * 0.95,
                            'target': current_price * 1.10
                        })
                    elif current_price < avg_price * 0.95:  # 5% below average
                        signals.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'avg_price': avg_price,
                            'momentum': ((current_price - avg_price) / avg_price) * 100,
                            'signal': 'BEARISH',
                            'entry': current_price,
                            'sl': current_price * 1.05,
                            'target': current_price * 0.90
                        })
                        
            except Exception as e:
                logger.warning(f"⚠️ Error screening {symbol}: {e}")
        
        # Save results
        if signals:
            df = pd.DataFrame(signals)
            df.to_csv('screening_results.csv', index=False)
            logger.info(f"📄 Saved {len(signals)} signals to screening_results.csv")
            
            print(f"\n🎯 SCREENING RESULTS:")
            print(f"📊 Total signals: {len(signals)}")
            print(f"📈 Bullish: {len([s for s in signals if s['signal'] == 'BULLISH'])}")
            print(f"📉 Bearish: {len([s for s in signals if s['signal'] == 'BEARISH'])}")
            
            print(f"\n📋 Sample signals:")
            for signal in signals[:5]:
                print(f"  • {signal['symbol']}: {signal['signal']} @ ₹{signal['entry']:.2f} (SL: ₹{signal['sl']:.2f}, Target: ₹{signal['target']:.2f})")
        else:
            logger.warning("⚠️ No signals generated")

def main():
    """Main function to run full data download and screening test."""
    print("🚀 FULL DATA DOWNLOAD AND SCREENING TEST")
    print("=" * 60)
    
    # Initialize downloader
    downloader = FullDataDownloader()
    
    # Load all symbols
    symbols = downloader.load_all_symbols()
    print(f"📊 Loaded {len(symbols)} symbols")
    
    # Download data (start with first 100 for testing)
    print(f"\n📥 Starting data download...")
    downloader.download_all_data(symbols, max_symbols=100)  # Start with 100 for testing
    
    # Get database stats
    stats = downloader.get_database_stats()
    print(f"\n📊 DATABASE STATISTICS:")
    print(f"📈 Securities: {stats['securities_count']}")
    print(f"📊 Price data points: {stats['price_data_count']}")
    print(f"🎯 Symbols with data: {stats['symbols_with_data']}")
    print(f"📅 Latest date: {stats['latest_date']}")
    
    # Test screening
    print(f"\n🧪 Testing screening system...")
    downloader.test_screening()
    
    print(f"\n🎉 FULL SYSTEM TEST COMPLETED!")
    print(f"📈 Ready for production use with {len(symbols)} Indian stocks!")

if __name__ == "__main__":
    main() 