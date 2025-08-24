#!/usr/bin/env python3
"""
Simple Fundamental Data Downloader
Downloads fundamental data using NSEUtility for all equities in database.
"""

import pandas as pd
import duckdb
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

from src.nsedata.NseUtility import NseUtils


class SimpleFundamentalDownloader:
    """Downloads fundamental data using NSEUtility."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        self.db_path = db_path
        self.nse = NseUtils()
        self.progress_file = "data/fundamental_progress.json"
        self.results_dir = "data/fundamental_data"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Download configuration
        self.max_workers = 5
        self.delay_between_requests = 1.0
        self.retry_attempts = 2
        
        # Initialize progress tracking
        self.progress = self._load_progress()
        
        print("SimpleFundamentalDownloader initialized")
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading progress: {e}")
        
        return {
            'start_time': datetime.now().isoformat(),
            'total_symbols': 0,
            'completed_symbols': [],
            'failed_symbols': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_progress(self):
        """Save progress to file."""
        self.progress['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def _init_database(self):
        """Initialize fundamental data table."""
        try:
            with duckdb.connect(self.db_path) as conn:
                # Create fundamental data table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fundamental_data (
                        symbol TEXT,
                        date TEXT,
                        market_cap REAL,
                        face_value REAL,
                        book_value REAL,
                        eps REAL,
                        pe_ratio REAL,
                        pb_ratio REAL,
                        dividend_yield REAL,
                        data_source TEXT,
                        last_updated TEXT,
                        PRIMARY KEY (symbol, date)
                    )
                """)
                
                # Create index for faster queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_symbol_date ON fundamental_data(symbol, date)")
                
                print("Fundamental data table initialized")
                
        except Exception as e:
            print(f"Database initialization failed: {e}")
            raise
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols from the database."""
        try:
            with duckdb.connect(self.db_path) as conn:
                symbols = pd.read_sql_query(
                    "SELECT DISTINCT symbol FROM price_data ORDER BY symbol", 
                    conn
                )['symbol'].tolist()
            
            print(f"Found {len(symbols)} symbols in database")
            return symbols
            
        except Exception as e:
            print(f"Error getting symbols: {e}")
            return []
    
    def _fetch_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamental data for a single symbol."""
        try:
            # Clean symbol (remove .NS if present)
            clean_symbol = symbol.replace('.NS', '')
            
            # Get equity info from NSEUtility
            equity_info = self.nse.equity_info(clean_symbol)
            
            if not equity_info or 'priceInfo' not in equity_info:
                return None
            
            price_info = equity_info['priceInfo']
            
            # Extract fundamental data
            fundamental_data = {
                'symbol': symbol,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'market_cap': price_info.get('marketCap', None),
                'face_value': price_info.get('faceValue', None),
                'book_value': price_info.get('bookValue', None),
                'eps': price_info.get('eps', None),
                'pe_ratio': price_info.get('pe', None),
                'pb_ratio': price_info.get('pb', None),
                'dividend_yield': price_info.get('dividendYield', None),
                'data_source': 'NSEUtility',
                'last_updated': datetime.now().isoformat()
            }
            
            # Check if we got any meaningful data
            non_none_values = [v for v in fundamental_data.values() if v is not None]
            if len(non_none_values) > 3:  # At least 3 non-None values
                return fundamental_data
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {e}")
            return None
    
    def _store_fundamental_data(self, fundamental_data: Dict[str, Any]):
        """Store fundamental data in database."""
        try:
            with duckdb.connect(self.db_path) as conn:
                # Insert or replace data
                conn.execute("""
                    INSERT OR REPLACE INTO fundamental_data 
                    (symbol, date, market_cap, face_value, book_value, eps, pe_ratio, pb_ratio, dividend_yield, data_source, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    fundamental_data.get('symbol'),
                    fundamental_data.get('date'),
                    fundamental_data.get('market_cap'),
                    fundamental_data.get('face_value'),
                    fundamental_data.get('book_value'),
                    fundamental_data.get('eps'),
                    fundamental_data.get('pe_ratio'),
                    fundamental_data.get('pb_ratio'),
                    fundamental_data.get('dividend_yield'),
                    fundamental_data.get('data_source'),
                    fundamental_data.get('last_updated')
                ])
                
        except Exception as e:
            print(f"Error storing fundamental data for {fundamental_data.get('symbol')}: {e}")
            raise
    
    def download_fundamental_data(self, max_symbols: Optional[int] = None):
        """Download fundamental data for all symbols."""
        try:
            print("🚀 Starting fundamental data download...")
            
            # Initialize database
            self._init_database()
            
            # Get all symbols
            symbols = self._get_all_symbols()
            if not symbols:
                print("❌ No symbols found in database")
                return
            
            # Limit symbols if specified
            if max_symbols:
                symbols = symbols[:max_symbols]
            
            self.progress['total_symbols'] = len(symbols)
            self.progress['start_time'] = datetime.now().isoformat()
            self._save_progress()
            
            print(f"📊 Downloading fundamental data for {len(symbols)} symbols")
            
            # Process symbols with threading
            completed = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self._process_single_symbol, symbol): symbol
                    for symbol in symbols
                }
                
                # Process completed tasks
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            completed += 1
                            self.progress['completed_symbols'].append(symbol)
                            print(f"✅ {symbol}: Fundamental data downloaded")
                        else:
                            failed += 1
                            self.progress['failed_symbols'].append({
                                'symbol': symbol,
                                'error': 'No fundamental data available',
                                'timestamp': datetime.now().isoformat()
                            })
                            print(f"⚠️ {symbol}: No fundamental data available")
                    except Exception as e:
                        failed += 1
                        self.progress['failed_symbols'].append({
                            'symbol': symbol,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        print(f"❌ {symbol}: {e}")
                    
                    # Update progress
                    if (completed + failed) % 10 == 0:
                        self._save_progress()
                        print(f"📊 Progress: {completed + failed}/{len(symbols)} (Completed: {completed}, Failed: {failed})")
                    
                    # Add delay to avoid rate limiting
                    time.sleep(self.delay_between_requests)
            
            # Final progress update
            self._save_progress()
            
            # Generate summary
            self._generate_summary(completed, failed, len(symbols))
            
            print(f"🎉 Fundamental data download completed!")
            print(f"   ✅ Completed: {completed}")
            print(f"   ❌ Failed: {failed}")
            print(f"   📊 Success Rate: {(completed/len(symbols)*100):.1f}%")
            
        except Exception as e:
            print(f"❌ Fundamental data download failed: {e}")
            raise
    
    def _process_single_symbol(self, symbol: str) -> bool:
        """Process a single symbol."""
        for attempt in range(self.retry_attempts):
            try:
                # Fetch fundamental data
                fundamental_data = self._fetch_fundamental_data(symbol)
                
                if fundamental_data:
                    # Store in database
                    self._store_fundamental_data(fundamental_data)
                    return True
                else:
                    return False
                    
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    print(f"⚠️ Retry {attempt + 1} for {symbol}: {e}")
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"❌ Failed after {self.retry_attempts} attempts for {symbol}: {e}")
                    return False
        
        return False
    
    def _generate_summary(self, completed: int, failed: int, total: int):
        """Generate download summary."""
        summary = {
            'download_date': datetime.now().isoformat(),
            'total_symbols': total,
            'completed_symbols': completed,
            'failed_symbols': failed,
            'success_rate': (completed / total * 100) if total > 0 else 0,
            'failed_symbols_list': [item['symbol'] for item in self.progress['failed_symbols']]
        }
        
        # Save summary
        summary_file = f"{self.results_dir}/fundamental_download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary saved to: {summary_file}")
    
    def get_fundamental_stats(self) -> Dict[str, Any]:
        """Get fundamental data statistics."""
        try:
            with duckdb.connect(self.db_path) as conn:
                # Get basic stats
                stats = {}
                
                # Total records
                result = conn.execute("SELECT COUNT(*) as total_records FROM fundamental_data").fetchone()
                stats['total_records'] = result[0] if result else 0
                
                # Unique symbols
                result = conn.execute("SELECT COUNT(DISTINCT symbol) as unique_symbols FROM fundamental_data").fetchone()
                stats['unique_symbols'] = result[0] if result else 0
                
                # Date range
                result = conn.execute("SELECT MIN(date) as start_date, MAX(date) as end_date FROM fundamental_data").fetchone()
                stats['date_range'] = {
                    'start_date': result[0] if result and result[0] else None,
                    'end_date': result[1] if result and result[1] else None
                }
                
                # Data completeness
                columns = ['pe_ratio', 'pb_ratio', 'market_cap', 'eps', 'dividend_yield']
                completeness = {}
                for col in columns:
                    result = conn.execute(f"SELECT COUNT(*) as count FROM fundamental_data WHERE {col} IS NOT NULL").fetchone()
                    completeness[col] = result[0] if result else 0
                
                stats['data_completeness'] = completeness
                
                return stats
                
        except Exception as e:
            print(f"Error getting fundamental stats: {e}")
            return {}


def main():
    """Main function to run the fundamental data downloader."""
    downloader = SimpleFundamentalDownloader()
    
    # Download fundamental data for all symbols (limit to 50 for testing)
    downloader.download_fundamental_data(max_symbols=50)
    
    # Print statistics
    stats = downloader.get_fundamental_stats()
    print("\n📊 FUNDAMENTAL DATA STATISTICS:")
    print(f"Total Records: {stats.get('total_records', 0)}")
    print(f"Unique Symbols: {stats.get('unique_symbols', 0)}")
    print(f"Date Range: {stats.get('date_range', {})}")
    print(f"Data Completeness: {stats.get('data_completeness', {})}")


if __name__ == "__main__":
    main()
