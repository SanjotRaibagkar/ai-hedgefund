#!/usr/bin/env python3
"""
NSE Fundamental Data Collector
Downloads corporate filings and financial results from NSE for all companies.
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass
import re

from src.nsedata.NseUtility import NseUtils
from src.data.database.duckdb_manager import DatabaseManager

@dataclass
class FundamentalData:
    """Data class for fundamental data."""
    symbol: str
    report_date: str
    period_type: str  # 'quarterly', 'half_yearly', 'annual'
    filing_date: str
    filing_type: str  # 'financial_results', 'corporate_filings'
    
    # Financial Metrics
    revenue: Optional[float] = None
    net_profit: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    operating_profit: Optional[float] = None
    ebitda: Optional[float] = None
    
    # Ratios
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    
    # Market Data
    market_cap: Optional[float] = None
    face_value: Optional[float] = None
    book_value: Optional[float] = None
    
    # Metadata
    source_url: Optional[str] = None
    raw_data: Optional[Dict] = None
    created_at: str = None
    updated_at: str = None

class NSEFundamentalCollector:
    """Collects fundamental data from NSE corporate filings and financial results."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        """Initialize the fundamental collector."""
        self.db_path = db_path
        self.nse = NseUtils()
        self.db_manager = DatabaseManager()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_workers = 3  # Conservative to avoid rate limiting
        self.delay_between_requests = 2.0  # 2 seconds between requests
        self.retry_attempts = 3
        self.session_timeout = 30
        
        # Progress tracking
        self.progress_file = "data/fundamental_progress.json"
        self.results_dir = "data/fundamental_data"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize progress
        self.progress = self._load_progress()
        
        # NSE URLs
        self.base_url = "https://www.nseindia.com"
        self.financial_results_url = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
        self.corporate_filings_url = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
        
        # Headers for NSE requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.logger.info("🚀 NSE Fundamental Collector initialized")
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading progress: {e}")
        
        return {
            'start_time': datetime.now().isoformat(),
            'total_symbols': 0,
            'completed_symbols': [],
            'failed_symbols': [],
            'last_updated': datetime.now().isoformat(),
            'current_batch': 0,
            'total_batches': 0
        }
    
    def _save_progress(self):
        """Save progress to file."""
        try:
            self.progress['last_updated'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")
    
    def _init_database(self):
        """Initialize fundamental data tables in database."""
        try:
            # Drop existing table if it exists to ensure correct schema
            drop_table_sql = "DROP TABLE IF EXISTS fundamental_data"
            
            # Create fundamental_data table with correct schema
            create_table_sql = """
            CREATE TABLE fundamental_data (
                symbol VARCHAR,
                report_date DATE,
                period_type VARCHAR,
                filing_date DATE,
                filing_type VARCHAR,
                revenue DOUBLE,
                net_profit DOUBLE,
                total_assets DOUBLE,
                total_liabilities DOUBLE,
                total_equity DOUBLE,
                operating_profit DOUBLE,
                ebitda DOUBLE,
                eps DOUBLE,
                pe_ratio DOUBLE,
                pb_ratio DOUBLE,
                roe DOUBLE,
                roa DOUBLE,
                debt_to_equity DOUBLE,
                market_cap DOUBLE,
                face_value DOUBLE,
                book_value DOUBLE,
                source_url VARCHAR,
                raw_data JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (symbol, report_date, period_type)
            )
            """
            
            conn = self.db_manager.connection
            conn.execute(drop_table_sql)
            conn.execute(create_table_sql)
            
            self.logger.info("✅ Fundamental data table initialized with correct schema")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols from the database."""
        try:
            # Create a fresh connection
            conn = self.db_manager.connection
            symbols = conn.execute(
                "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
            ).fetchdf()['symbol'].tolist()
            
            self.logger.info(f"📊 Found {len(symbols)} symbols in database")
            return symbols
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbols: {e}")
            # Fallback to a few test symbols if database query fails
            fallback_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
            self.logger.info(f"📊 Using fallback symbols: {fallback_symbols}")
            return fallback_symbols
    
    def _make_nse_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to NSE with proper session handling."""
        try:
            # Create a new session for each request
            session = requests.Session()
            session.headers.update(self.headers)
            
            # First, get the main page to establish session
            session.get(self.base_url, timeout=self.session_timeout)
            time.sleep(1)  # Small delay
            
            # Make the actual request
            response = session.get(url, params=params, timeout=self.session_timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"⚠️ Request failed for {url}: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Request error for {url}: {e}")
            return None
    
    def _get_financial_results(self, symbol: str) -> List[FundamentalData]:
        """Get financial results for a symbol using NSEUtility."""
        try:
            # Clean symbol
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # Use NSEUtility to get equity info (includes some fundamental data)
            equity_info = self.nse.equity_info(clean_symbol)
            if not equity_info or 'priceInfo' not in equity_info:
                return []
            
            price_info = equity_info['priceInfo']
            
            # Extract available fundamental data from equity info
            fundamental_data = []
            
            # Create fundamental data record from available info
            financial_data = FundamentalData(
                symbol=symbol,
                report_date=datetime.now().strftime('%Y-%m-%d'),
                period_type='current',
                filing_date=datetime.now().strftime('%Y-%m-%d'),
                filing_type='equity_info',
                eps=self._extract_numeric(price_info.get('eps')),
                pe_ratio=self._extract_numeric(price_info.get('pe')),
                pb_ratio=self._extract_numeric(price_info.get('pb')),
                market_cap=self._extract_numeric(price_info.get('marketCap')),
                face_value=self._extract_numeric(price_info.get('faceValue')),
                book_value=self._extract_numeric(price_info.get('bookValue')),
                source_url=f"https://www.nseindia.com/get-quotes/equity?symbol={clean_symbol}",
                raw_data=equity_info,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            fundamental_data.append(financial_data)
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting financial results for {symbol}: {e}")
            return []
    
    def _get_corporate_filings(self, symbol: str) -> List[FundamentalData]:
        """Get corporate filings for a symbol using NSEUtility."""
        try:
            # Clean symbol
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # Get corporate actions for the last 30 days
            corporate_actions = self.nse.get_corporate_action(from_date_str=None, to_date_str=None)
            if corporate_actions is None or corporate_actions.empty:
                return []
            
            # Filter corporate actions for this symbol
            symbol_actions = corporate_actions[
                corporate_actions['symbol'].str.contains(clean_symbol, case=False, na=False)
            ]
            
            fundamental_data = []
            
            # Process corporate actions
            for _, action in symbol_actions.iterrows():
                try:
                    # Extract action information
                    subject = action.get('subject', '')
                    ex_date = action.get('exDate', '')
                    record_date = action.get('recordDate', '')
                    
                    # Only process financial-related actions
                    if self._is_financial_filing(subject):
                        corporate_data = FundamentalData(
                            symbol=symbol,
                            report_date=ex_date if ex_date else record_date,
                            period_type='corporate_action',
                            filing_date=record_date,
                            filing_type='corporate_actions',
                            source_url="https://www.nseindia.com/companies-listing/corporate-filings-actions",
                            raw_data=action.to_dict(),
                            created_at=datetime.now().isoformat(),
                            updated_at=datetime.now().isoformat()
                        )
                        
                        fundamental_data.append(corporate_data)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing corporate action for {symbol}: {e}")
                    continue
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting corporate filings for {symbol}: {e}")
            return []
    
    def _determine_period_type(self, period: str) -> str:
        """Determine the period type from period string."""
        period_lower = period.lower()
        
        if 'quarter' in period_lower or 'q' in period_lower:
            return 'quarterly'
        elif 'half' in period_lower or 'semi' in period_lower:
            return 'half_yearly'
        elif 'annual' in period_lower or 'year' in period_lower:
            return 'annual'
        else:
            return 'unknown'
    
    def _extract_numeric(self, value: Any) -> Optional[float]:
        """Extract numeric value from various formats."""
        if value is None:
            return None
        
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Remove common non-numeric characters
                cleaned = re.sub(r'[^\d.-]', '', value)
                if cleaned:
                    return float(cleaned)
            return None
        except (ValueError, TypeError):
            return None
    
    def _is_financial_filing(self, filing_type: str) -> bool:
        """Check if filing type is financial-related."""
        financial_keywords = [
            'financial', 'results', 'quarterly', 'annual', 'balance', 'profit', 
            'loss', 'revenue', 'earnings', 'dividend', 'bonus', 'split'
        ]
        
        filing_lower = filing_type.lower()
        return any(keyword in filing_lower for keyword in financial_keywords)
    
    def _fetch_symbol_data(self, symbol: str) -> Tuple[str, List[FundamentalData]]:
        """Fetch all fundamental data for a single symbol."""
        try:
            self.logger.info(f"📊 Fetching fundamental data for {symbol}")
            
            all_data = []
            
            # Get financial results (equity info)
            financial_results = self._get_financial_results(symbol)
            all_data.extend(financial_results)
            
            # Get corporate filings (corporate actions)
            corporate_filings = self._get_corporate_filings(symbol)
            all_data.extend(corporate_filings)
            
            # Add delay to avoid rate limiting
            time.sleep(self.delay_between_requests)
            
            self.logger.info(f"✅ Fetched {len(all_data)} records for {symbol}")
            return symbol, all_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return symbol, []
    
    def get_upcoming_results(self) -> List[FundamentalData]:
        """Get upcoming financial results calendar."""
        try:
            self.logger.info("📅 Fetching upcoming financial results calendar...")
            
            # Get upcoming results calendar
            upcoming_results = self.nse.get_upcoming_results_calendar()
            if upcoming_results is None or upcoming_results.empty:
                return []
            
            fundamental_data = []
            
            # Process upcoming results
            for _, result in upcoming_results.iterrows():
                try:
                    # Extract result information
                    symbol = result.get('symbol', '')
                    purpose = result.get('purpose', '')
                    ex_date = result.get('exDate', '')
                    record_date = result.get('recordDate', '')
                    
                    # Create fundamental data record
                    result_data = FundamentalData(
                        symbol=symbol,
                        report_date=ex_date if ex_date else record_date,
                        period_type='upcoming_results',
                        filing_date=record_date,
                        filing_type='upcoming_results',
                        source_url="https://www.nseindia.com/companies-listing/corporate-filings-event-calendar",
                        raw_data=result.to_dict(),
                        created_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat()
                    )
                    
                    fundamental_data.append(result_data)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing upcoming result: {e}")
                    continue
            
            self.logger.info(f"✅ Fetched {len(fundamental_data)} upcoming results")
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting upcoming results: {e}")
            return []
    
    def _save_to_database(self, fundamental_data: List[FundamentalData]):
        """Save fundamental data to database."""
        try:
            if not fundamental_data:
                return
            
            # Convert to DataFrame
            data_list = []
            for data in fundamental_data:
                data_dict = {
                    'symbol': data.symbol,
                    'report_date': data.report_date,
                    'period_type': data.period_type,
                    'filing_date': data.filing_date,
                    'filing_type': data.filing_type,
                    'revenue': data.revenue,
                    'net_profit': data.net_profit,
                    'total_assets': data.total_assets,
                    'total_liabilities': data.total_liabilities,
                    'total_equity': data.total_equity,
                    'operating_profit': data.operating_profit,
                    'ebitda': data.ebitda,
                    'eps': data.eps,
                    'pe_ratio': data.pe_ratio,
                    'pb_ratio': data.pb_ratio,
                    'roe': data.roe,
                    'roa': data.roa,
                    'debt_to_equity': data.debt_to_equity,
                    'market_cap': data.market_cap,
                    'face_value': data.face_value,
                    'book_value': data.book_value,
                    'source_url': data.source_url,
                    'raw_data': json.dumps(data.raw_data) if data.raw_data else None,
                    'created_at': data.created_at,
                    'updated_at': data.updated_at
                }
                data_list.append(data_dict)
            
            df = pd.DataFrame(data_list)
            
            # Insert into database
            conn = self.db_manager.connection
            # Use INSERT OR REPLACE to handle duplicates
            for _, row in df.iterrows():
                insert_sql = """
                INSERT OR REPLACE INTO fundamental_data (
                    symbol, report_date, period_type, filing_date, filing_type,
                    revenue, net_profit, total_assets, total_liabilities, total_equity,
                    operating_profit, ebitda, eps, pe_ratio, pb_ratio, roe, roa,
                    debt_to_equity, market_cap, face_value, book_value,
                    source_url, raw_data, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                conn.execute(insert_sql, [
                    row['symbol'], row['report_date'], row['period_type'], 
                    row['filing_date'], row['filing_type'], row['revenue'],
                    row['net_profit'], row['total_assets'], row['total_liabilities'],
                    row['total_equity'], row['operating_profit'], row['ebitda'],
                    row['eps'], row['pe_ratio'], row['pb_ratio'], row['roe'],
                    row['roa'], row['debt_to_equity'], row['market_cap'],
                    row['face_value'], row['book_value'], row['source_url'],
                    row['raw_data'], row['created_at'], row['updated_at']
                ])
            
            self.logger.info(f"💾 Saved {len(df)} fundamental records to database")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving to database: {e}")
    
    def download_all_fundamentals(self, batch_size: int = 50):
        """Download fundamental data for all symbols in batches."""
        try:
            self.logger.info("🚀 Starting fundamental data download for all symbols")
            
            # Initialize database
            self._init_database()
            
            # Get all symbols
            symbols = self._get_all_symbols()
            if not symbols:
                self.logger.error("❌ No symbols found in database")
                return
            
            # Update progress
            self.progress['total_symbols'] = len(symbols)
            self.progress['total_batches'] = (len(symbols) + batch_size - 1) // batch_size
            self._save_progress()
            
            self.logger.info(f"📊 Processing {len(symbols)} symbols in {self.progress['total_batches']} batches")
            
            # Process in batches
            for batch_num in range(self.progress['total_batches']):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(symbols))
                batch_symbols = symbols[start_idx:end_idx]
                
                self.logger.info(f"📦 Processing batch {batch_num + 1}/{self.progress['total_batches']}: {len(batch_symbols)} symbols")
                
                # Process batch with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks
                    future_to_symbol = {
                        executor.submit(self._fetch_symbol_data, symbol): symbol 
                        for symbol in batch_symbols
                    }
                    
                    # Process completed tasks
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            symbol, fundamental_data = future.result()
                            
                            if fundamental_data:
                                self._save_to_database(fundamental_data)
                                self.progress['completed_symbols'].append(symbol)
                            else:
                                self.progress['failed_symbols'].append(symbol)
                            
                        except Exception as e:
                            self.logger.error(f"❌ Error processing {symbol}: {e}")
                            self.progress['failed_symbols'].append(symbol)
                
                # Update progress
                self.progress['current_batch'] = batch_num + 1
                self._save_progress()
                
                # Batch completion summary
                completed = len(self.progress['completed_symbols'])
                failed = len(self.progress['failed_symbols'])
                self.logger.info(f"📊 Batch {batch_num + 1} completed: {completed} success, {failed} failed")
                
                # Delay between batches
                if batch_num < self.progress['total_batches'] - 1:
                    time.sleep(5)  # 5 second delay between batches
            
            # Final summary
            self.logger.info("🎉 Fundamental data download completed!")
            self.logger.info(f"✅ Successfully processed: {len(self.progress['completed_symbols'])} symbols")
            self.logger.info(f"❌ Failed: {len(self.progress['failed_symbols'])} symbols")
            
            # Save final progress
            self._save_progress()
            
        except Exception as e:
            self.logger.error(f"❌ Error in download_all_fundamentals: {e}")
            self._save_progress()
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get current download status."""
        return {
            'total_symbols': self.progress['total_symbols'],
            'completed_symbols': len(self.progress['completed_symbols']),
            'failed_symbols': len(self.progress['failed_symbols']),
            'current_batch': self.progress['current_batch'],
            'total_batches': self.progress['total_batches'],
            'progress_percentage': (len(self.progress['completed_symbols']) / self.progress['total_symbols'] * 100) if self.progress['total_symbols'] > 0 else 0,
            'last_updated': self.progress['last_updated']
        }


def main():
    """Main function for testing."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Initialize collector
    collector = NSEFundamentalCollector()
    
    # Download all fundamental data
    collector.download_all_fundamentals(batch_size=20)  # Smaller batch size for testing
    
    # Print status
    status = collector.get_download_status()
    print(f"\n📊 Download Status:")
    print(f"   Total Symbols: {status['total_symbols']}")
    print(f"   Completed: {status['completed_symbols']}")
    print(f"   Failed: {status['failed_symbols']}")
    print(f"   Progress: {status['progress_percentage']:.1f}%")


if __name__ == "__main__":
    main()
