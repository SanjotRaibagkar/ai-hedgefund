"""
Database Provider for accessing comprehensive equity data from DuckDB/SQLite databases.
"""

import pandas as pd
import duckdb
import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger

from src.data.models import Price, FinancialMetrics


class DatabaseProvider:
    """Provider for accessing comprehensive equity data from databases."""
    
    def __init__(self, database_path: str = None):
        """
        Initialize database provider.
        
        Args:
            database_path: Path to the database file. If None, uses the most comprehensive database.
        """
        self.database_path = database_path or self._get_best_database()
        self.db_type = 'duckdb' if self.database_path.endswith('.duckdb') else 'sqlite'
        logger.info(f"DatabaseProvider initialized with {self.database_path} ({self.db_type})")
    
    def _get_best_database(self) -> str:
        """Get the database with the most comprehensive data."""
        data_dir = 'data'
        databases = []
        
        for file in os.listdir(data_dir):
            if file.endswith(('.db', '.duckdb')):
                db_path = os.path.join(data_dir, file)
                try:
                    if file.endswith('.duckdb'):
                        conn = duckdb.connect(db_path)
                    else:
                        conn = sqlite3.connect(db_path)
                    
                    # Check if price_data table exists
                    if self.db_type == 'duckdb':
                        tables = conn.execute("SHOW TABLES").fetchall()
                    else:
                        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    
                    table_names = [table[0] for table in tables]
                    
                    if 'price_data' in table_names:
                        # Get record count
                        count = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
                        databases.append((db_path, count))
                    
                    conn.close()
                    
                except Exception as e:
                    logger.warning(f"Could not check {db_path}: {e}")
        
        if not databases:
            raise ValueError("No valid databases found with price_data table")
        
        # Sort by record count (descending) and return the best one
        databases.sort(key=lambda x: x[1], reverse=True)
        best_db = databases[0][0]
        logger.info(f"Selected database: {best_db} with {databases[0][1]} records")
        return best_db
    
    def get_prices(self, symbol: str, start_date: str, end_date: str) -> List[Price]:
        """Get historical price data from database."""
        try:
            # Clean the symbol (remove .NS suffix)
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('.NSE', '').replace('.BSE', '')
            
            if self.db_type == 'duckdb':
                conn = duckdb.connect(self.database_path)
            else:
                conn = sqlite3.connect(self.database_path)
            
            # Query the database
            query = """
                SELECT date, open_price, high_price, low_price, close_price, volume, turnover
                FROM price_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=[clean_symbol, start_date, end_date])
            conn.close()
            
            if df.empty:
                logger.warning(f"No price data found for {symbol} from {start_date} to {end_date}")
                return []
            
            # Convert to Price objects
            prices = []
            for _, row in df.iterrows():
                price = Price(
                    date=row['date'],
                    open_price=row['open_price'],
                    high_price=row['high_price'],
                    low_price=row['low_price'],
                    close_price=row['close_price'],
                    volume=row['volume'],
                    turnover=row.get('turnover', 0)
                )
                prices.append(price)
            
            logger.info(f"Retrieved {len(prices)} price records for {symbol}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching prices for {symbol}: {e}")
            return []
    
    def get_prices_as_dataframe(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data as DataFrame for screening modules."""
        try:
            # Clean the symbol (remove .NS suffix)
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('.NSE', '').replace('.BSE', '')
            
            if self.db_type == 'duckdb':
                conn = duckdb.connect(self.database_path)
            else:
                conn = sqlite3.connect(self.database_path)
            
            # Query the database
            query = """
                SELECT date, open_price, high_price, low_price, close_price, volume, turnover
                FROM price_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=[clean_symbol, start_date, end_date])
            conn.close()
            
            if df.empty:
                logger.warning(f"No price data found for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Ensure proper column names
            df = df.rename(columns={
                'open_price': 'open_price',
                'high_price': 'high_price',
                'low_price': 'low_price',
                'close_price': 'close_price',
                'volume': 'volume'
            })
            
            logger.info(f"Retrieved {len(df)} price records for {symbol} as DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching prices as DataFrame for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_financial_metrics(self, symbol: str, end_date: str, period: str = "ttm", limit: int = 1) -> List[FinancialMetrics]:
        """Get financial metrics from database if available."""
        try:
            # Clean the symbol
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('.NSE', '').replace('.BSE', '')
            
            if self.db_type == 'duckdb':
                conn = duckdb.connect(self.database_path)
            else:
                conn = sqlite3.connect(self.database_path)
            
            # Check if fundamental_data table exists
            if self.db_type == 'duckdb':
                tables = conn.execute("SHOW TABLES").fetchall()
            else:
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            table_names = [table[0] for table in tables]
            
            if 'fundamental_data' not in table_names:
                logger.debug(f"No fundamental_data table found in {self.database_path}")
                conn.close()
                return []
            
            # Query fundamental data
            query = """
                SELECT * FROM fundamental_data 
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[clean_symbol, end_date, limit])
            conn.close()
            
            if df.empty:
                logger.debug(f"No fundamental data found for {symbol}")
                return []
            
            # Convert to FinancialMetrics objects
            metrics = []
            for _, row in df.iterrows():
                metric = FinancialMetrics(
                    report_period=row['date'],
                    period=period,
                    currency="INR",
                    market_cap=row.get('market_cap'),
                    enterprise_value=row.get('market_cap'),  # Approximate
                    price_to_earnings_ratio=row.get('pe_ratio'),
                    price_to_book_ratio=row.get('pb_ratio'),
                    return_on_equity=row.get('roe'),
                    debt_to_equity=row.get('debt_to_equity'),
                    earnings_growth=None,  # Not available in current schema
                    revenue_growth=None,
                    book_value_growth=None
                )
                metrics.append(metric)
            
            logger.info(f"Retrieved {len(metrics)} fundamental metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return []
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all available symbols in the database."""
        try:
            if self.db_type == 'duckdb':
                conn = duckdb.connect(self.database_path)
            else:
                conn = sqlite3.connect(self.database_path)
            
            query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            symbols = df['symbol'].tolist()
            logger.info(f"Found {len(symbols)} symbols in database")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            if self.db_type == 'duckdb':
                conn = duckdb.connect(self.database_path)
            else:
                conn = sqlite3.connect(self.database_path)
            
            # Get basic stats
            stats = {
                'database_path': self.database_path,
                'database_type': self.db_type
            }
            
            # Price data stats
            price_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as total_symbols,
                    MIN(date) as start_date,
                    MAX(date) as end_date
                FROM price_data
            """).fetchone()
            
            stats['price_data'] = {
                'total_records': price_stats[0],
                'total_symbols': price_stats[1],
                'start_date': price_stats[2],
                'end_date': price_stats[3]
            }
            
            # Check if fundamental data exists
            if self.db_type == 'duckdb':
                tables = conn.execute("SHOW TABLES").fetchall()
            else:
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            table_names = [table[0] for table in tables]
            
            if 'fundamental_data' in table_names:
                fund_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT symbol) as total_symbols
                    FROM fundamental_data
                """).fetchone()
                
                stats['fundamental_data'] = {
                    'total_records': fund_stats[0],
                    'total_symbols': fund_stats[1]
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
