"""
DuckDB Provider for Comprehensive Historical Data

This provider uses the comprehensive DuckDB database to fetch historical data
instead of making live API calls, providing faster and more reliable access.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import duckdb

from src.data.providers.base_provider import BaseDataProvider, DataProviderError

logger = logging.getLogger(__name__)


class DuckDBProvider(BaseDataProvider):
    """DuckDB-based data provider for comprehensive historical data."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        """Initialize DuckDB provider."""
        try:
            self.db_path = db_path
            self._cache = {}
            self._cache_ttl = 300  # 5 minutes cache
            logger.info(f"DuckDB provider initialized with database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB provider: {e}")
            raise DataProviderError(f"DuckDB initialization failed: {e}")
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._cache_ttl):
                return data
            else:
                del self._cache[key]
        return None
    
    def _set_cached_data(self, key: str, data: Any):
        """Set data in cache with timestamp."""
        self._cache[key] = (data, datetime.now())
    
    def get_prices(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Get historical price data for a symbol from DuckDB."""
        try:
            cache_key = f"prices_{symbol}_{start_date}_{end_date}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with duckdb.connect(self.db_path) as conn:
                query = """
                SELECT symbol, date, open_price, high_price, low_price, close_price, volume, turnover
                FROM price_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """
                
                result = conn.execute(query, [symbol, start_date, end_date]).fetchdf()
                
                if result.empty:
                    logger.warning(f"No price data found for {symbol} between {start_date} and {end_date}")
                    return []
                
                # Convert to list of dictionaries
                prices = []
                for _, row in result.iterrows():
                    price = {
                        'symbol': row['symbol'],
                        'date': row['date'],
                        'open': row['open_price'],
                        'high': row['high_price'],
                        'low': row['low_price'],
                        'close': row['close_price'],
                        'volume': row['volume'],
                        'turnover': row.get('turnover', 0)
                    }
                    prices.append(price)
                
                self._set_cached_data(cache_key, prices)
                logger.info(f"Retrieved {len(prices)} price records for {symbol}")
                return prices
                
        except Exception as e:
            logger.error(f"Error fetching prices for {symbol}: {e}")
            return []
    
    def get_prices_as_dataframe(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data as DataFrame for screening modules."""
        try:
            cache_key = f"df_{symbol}_{start_date}_{end_date}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with duckdb.connect(self.db_path) as conn:
                query = """
                SELECT symbol, date, open_price, high_price, low_price, close_price, volume, turnover
                FROM price_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """
                
                df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()
                
                if df.empty:
                    logger.warning(f"No price data found for {symbol} between {start_date} and {end_date}")
                    return pd.DataFrame()
                
                # Ensure column names match what screening modules expect
                df = df.rename(columns={
                    'open_price': 'open_price',
                    'high_price': 'high_price', 
                    'low_price': 'low_price',
                    'close_price': 'close_price',
                    'volume': 'volume'
                })
                
                self._set_cached_data(cache_key, df)
                logger.info(f"Retrieved {len(df)} price records for {symbol} as DataFrame")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching prices as DataFrame for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols from DuckDB."""
        try:
            with duckdb.connect(self.db_path) as conn:
                query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
                result = conn.execute(query).fetchdf()
                symbols = result['symbol'].tolist()
                logger.info(f"Retrieved {len(symbols)} symbols from DuckDB")
                return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def get_symbol_count(self) -> int:
        """Get total number of symbols in database."""
        try:
            with duckdb.connect(self.db_path) as conn:
                query = "SELECT COUNT(DISTINCT symbol) as count FROM price_data"
                result = conn.execute(query).fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting symbol count: {e}")
            return 0
    
    def get_data_range(self) -> Dict[str, str]:
        """Get the date range of available data."""
        try:
            with duckdb.connect(self.db_path) as conn:
                query = "SELECT MIN(date) as start_date, MAX(date) as end_date FROM price_data"
                result = conn.execute(query).fetchone()
                if result:
                    return {
                        'start_date': result[0],
                        'end_date': result[1]
                    }
                return {}
        except Exception as e:
            logger.error(f"Error getting data range: {e}")
            return {}

    # Required abstract methods from BaseDataProvider
    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10):
        """Get financial metrics for a ticker."""
        logger.warning(f"Financial metrics not available in DuckDB provider for {ticker}")
        return []
    
    def get_line_items(self, ticker: str, line_items: List[str], end_date: str, period: str = "ttm", limit: int = 10):
        """Get specific financial line items for a ticker."""
        logger.warning(f"Line items not available in DuckDB provider for {ticker}")
        return []
    
    def get_market_cap(self, ticker: str, end_date: str):
        """Get market capitalization for a ticker."""
        logger.warning(f"Market cap not available in DuckDB provider for {ticker}")
        return None
    
    def get_company_news(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000):
        """Get company news for a ticker."""
        logger.warning(f"Company news not available in DuckDB provider for {ticker}")
        return []
    
    def get_insider_trades(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000):
        """Get insider trades for a ticker."""
        logger.warning(f"Insider trades not available in DuckDB provider for {ticker}")
        return []
    
    def supports_ticker(self, ticker: str) -> bool:
        """Check if this provider supports the given ticker format."""
        # Support Indian tickers (NSE format)
        ticker = ticker.upper()
        return any(ticker.endswith(suffix) for suffix in ['.NS', '.BO', '.NSE', '.BSE']) or '.' not in ticker
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "DuckDB"
