#!/usr/bin/env python3
"""
EOD Data Provider
Provides access to the four EOD tables for ML models and screeners.
"""

import pandas as pd
import duckdb
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FNOBhavCopy:
    """FNO Bhav Copy data model."""
    fin_instrm_id: str
    tckr_symb: str
    strk_pric: float
    optn_tp: str
    xpry_dt: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int
    change: float
    change_percent: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    underlying: str


@dataclass
class EquityBhavCopyDelivery:
    """Equity Bhav Copy with Delivery data model."""
    symbol: str
    series: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    delivery_qty: int
    delivery_per: float
    turnover: float


@dataclass
class BhavCopyIndices:
    """Bhav Copy Indices data model."""
    index_name: str
    trade_date: str
    open_index_value: float
    high_index_value: float
    low_index_value: float
    closing_index_value: float
    points_change: float
    change_percent: float
    volume: int
    turnover_rs_cr: float
    pe_ratio: float
    pb_ratio: float
    div_yield: float


@dataclass
class FIIDIIActivity:
    """FII DII Activity data model."""
    category: str
    date: str
    buy_value: float
    sell_value: float
    net_value: float
    trade_date: str


class EODDataProvider:
    """Provider for EOD (End of Day) data from the four new tables."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to the DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"✅ Connected to EOD database: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to EOD database: {e}")
            self.conn = None
    
    def _ensure_connection(self):
        """Ensure database connection is active."""
        if self.conn is None:
            self._connect()
    
    def get_fno_bhav_copy(
        self, 
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get FNO Bhav Copy data.
        
        Args:
            symbol: Filter by symbol (e.g., 'NIFTY', 'BANKNIFTY')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with FNO Bhav Copy data
        """
        try:
            self._ensure_connection()
            if self.conn is None:
                return pd.DataFrame()
            
            query = "SELECT * FROM fno_bhav_copy WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND TckrSymb LIKE ?"
                params.append(f"%{symbol}%")
            
            if start_date:
                query += " AND TRADE_DATE >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND TRADE_DATE <= ?"
                params.append(end_date)
            
            query += " ORDER BY TRADE_DATE DESC, TckrSymb"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = self.conn.execute(query, params).df()
            logger.info(f"✅ Retrieved {len(df)} FNO Bhav Copy records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching FNO Bhav Copy data: {e}")
            return pd.DataFrame()
    
    def get_equity_bhav_copy_delivery(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get Equity Bhav Copy with Delivery data.
        
        Args:
            symbol: Filter by symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with Equity Bhav Copy Delivery data
        """
        try:
            self._ensure_connection()
            if self.conn is None:
                return pd.DataFrame()
            
            query = "SELECT * FROM equity_bhav_copy_delivery WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND SYMBOL LIKE ?"
                params.append(f"%{symbol}%")
            
            if start_date:
                query += " AND TRADE_DATE >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND TRADE_DATE <= ?"
                params.append(end_date)
            
            query += " ORDER BY TRADE_DATE DESC, SYMBOL"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = self.conn.execute(query, params).df()
            logger.info(f"✅ Retrieved {len(df)} Equity Bhav Copy Delivery records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching Equity Bhav Copy Delivery data: {e}")
            return pd.DataFrame()
    
    def get_bhav_copy_indices(
        self,
        index_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get Bhav Copy Indices data.
        
        Args:
            index_name: Filter by index name (e.g., 'NIFTY 50', 'NIFTY BANK')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with Bhav Copy Indices data
        """
        try:
            self._ensure_connection()
            if self.conn is None:
                return pd.DataFrame()
            
            query = "SELECT * FROM bhav_copy_indices WHERE 1=1"
            params = []
            
            if index_name:
                query += " AND Index_Name LIKE ?"
                params.append(f"%{index_name}%")
            
            if start_date:
                query += " AND TRADE_DATE >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND TRADE_DATE <= ?"
                params.append(end_date)
            
            query += " ORDER BY TRADE_DATE DESC, Index_Name"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = self.conn.execute(query, params).df()
            logger.info(f"✅ Retrieved {len(df)} Bhav Copy Indices records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching Bhav Copy Indices data: {e}")
            return pd.DataFrame()
    
    def get_fii_dii_activity(
        self,
        category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get FII DII Activity data.
        
        Args:
            category: Filter by category ('FII' or 'DII')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with FII DII Activity data
        """
        try:
            self._ensure_connection()
            if self.conn is None:
                return pd.DataFrame()
            
            query = "SELECT * FROM fii_dii_activity WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            if start_date:
                query += " AND TRADE_DATE >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND TRADE_DATE <= ?"
                params.append(end_date)
            
            query += " ORDER BY TRADE_DATE DESC, category"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = self.conn.execute(query, params).df()
            logger.info(f"✅ Retrieved {len(df)} FII DII Activity records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching FII DII Activity data: {e}")
            return pd.DataFrame()
    
    def get_latest_fno_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get latest FNO data for a specific symbol."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.get_fno_bhav_copy(symbol=symbol, start_date=start_date, end_date=end_date)
    
    def get_latest_equity_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get latest equity data for a specific symbol."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.get_equity_bhav_copy_delivery(symbol=symbol, start_date=start_date, end_date=end_date)
    
    def get_latest_index_data(self, index_name: str, days: int = 30) -> pd.DataFrame:
        """Get latest index data for a specific index."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.get_bhav_copy_indices(index_name=index_name, start_date=start_date, end_date=end_date)
    
    def get_latest_fii_dii_data(self, days: int = 30) -> pd.DataFrame:
        """Get latest FII/DII activity data."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.get_fii_dii_activity(start_date=start_date, end_date=end_date)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for all EOD tables."""
        try:
            self._ensure_connection()
            if self.conn is None:
                return {}
            
            stats = {}
            
            # FNO Bhav Copy stats
            fno_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date,
                       COUNT(DISTINCT TckrSymb) as unique_symbols
                FROM fno_bhav_copy
            """).df()
            
            if not fno_stats.empty:
                stats['fno_bhav_copy'] = fno_stats.iloc[0].to_dict()
            
            # Equity Bhav Copy Delivery stats
            equity_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date,
                       COUNT(DISTINCT SYMBOL) as unique_symbols
                FROM equity_bhav_copy_delivery
            """).df()
            
            if not equity_stats.empty:
                stats['equity_bhav_copy_delivery'] = equity_stats.iloc[0].to_dict()
            
            # Bhav Copy Indices stats
            indices_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date,
                       COUNT(DISTINCT Index_Name) as unique_indices
                FROM bhav_copy_indices
            """).df()
            
            if not indices_stats.empty:
                stats['bhav_copy_indices'] = indices_stats.iloc[0].to_dict()
            
            # FII DII Activity stats
            fii_stats = self.conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT TRADE_DATE) as unique_dates,
                       MIN(TRADE_DATE) as earliest_date,
                       MAX(TRADE_DATE) as latest_date,
                       COUNT(DISTINCT category) as unique_categories
                FROM fii_dii_activity
            """).df()
            
            if not fii_stats.empty:
                stats['fii_dii_activity'] = fii_stats.iloc[0].to_dict()
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}
    
    def get_top_symbols_by_volume(self, table: str = 'fno_bhav_copy', days: int = 30, limit: int = 10) -> pd.DataFrame:
        """Get top symbols by volume for the specified table."""
        try:
            self._ensure_connection()
            if self.conn is None:
                return pd.DataFrame()
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            if table == 'fno_bhav_copy':
                query = """
                    SELECT TckrSymb, SUM(TtlTradgVol) as total_volume, 
                           COUNT(DISTINCT TRADE_DATE) as trading_days,
                           AVG(ClsPric) as avg_price
                    FROM fno_bhav_copy 
                    WHERE TRADE_DATE >= ? AND TRADE_DATE <= ?
                    GROUP BY TckrSymb 
                    ORDER BY total_volume DESC 
                    LIMIT ?
                """
            elif table == 'equity_bhav_copy_delivery':
                query = """
                    SELECT SYMBOL, SUM(TTL_TRD_QNTY) as total_volume,
                           COUNT(DISTINCT TRADE_DATE) as trading_days,
                           AVG(CLOSE_PRICE) as avg_price
                    FROM equity_bhav_copy_delivery 
                    WHERE TRADE_DATE >= ? AND TRADE_DATE <= ?
                    GROUP BY SYMBOL 
                    ORDER BY total_volume DESC 
                    LIMIT ?
                """
            else:
                logger.error(f"❌ Unsupported table for volume analysis: {table}")
                return pd.DataFrame()
            
            df = self.conn.execute(query, [start_date, end_date, limit]).df()
            logger.info(f"✅ Retrieved top {len(df)} symbols by volume from {table}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting top symbols by volume: {e}")
            return pd.DataFrame()
    
    def get_market_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get market summary for a specific date or latest available date."""
        try:
            self._ensure_connection()
            if self.conn is None:
                return {}
            
            if date is None:
                # Get the latest available date
                latest_date = self.conn.execute("""
                    SELECT MAX(TRADE_DATE) as latest_date FROM fno_bhav_copy
                """).df()
                if latest_date.empty:
                    return {}
                date = latest_date.iloc[0]['latest_date']
            
            summary = {
                'date': date,
                'fno_summary': {},
                'equity_summary': {},
                'indices_summary': {},
                'fii_dii_summary': {}
            }
            
            # FNO Summary
            fno_summary = self.conn.execute("""
                SELECT COUNT(*) as total_contracts,
                       COUNT(DISTINCT TckrSymb) as unique_symbols,
                       SUM(TtlTradgVol) as total_volume,
                       SUM(OpnIntrst) as total_oi
                FROM fno_bhav_copy 
                WHERE TRADE_DATE = ?
            """, [date]).df()
            
            if not fno_summary.empty:
                summary['fno_summary'] = fno_summary.iloc[0].to_dict()
            
            # Equity Summary
            equity_summary = self.conn.execute("""
                SELECT COUNT(*) as total_stocks,
                       SUM(TTL_TRD_QNTY) as total_volume,
                       SUM(DELIV_QTY) as total_delivery_qty,
                       AVG(DELIV_PER) as avg_delivery_percent
                FROM equity_bhav_copy_delivery 
                WHERE TRADE_DATE = ?
            """, [date]).df()
            
            if not equity_summary.empty:
                summary['equity_summary'] = equity_summary.iloc[0].to_dict()
            
            # Indices Summary
            indices_summary = self.conn.execute("""
                SELECT COUNT(*) as total_indices,
                       AVG(Points_Change) as avg_points_change,
                       AVG(Change_Percent) as avg_change_percent
                FROM bhav_copy_indices 
                WHERE TRADE_DATE = ?
            """, [date]).df()
            
            if not indices_summary.empty:
                summary['indices_summary'] = indices_summary.iloc[0].to_dict()
            
            # FII DII Summary
            fii_dii_summary = self.conn.execute("""
                SELECT category,
                       SUM(buyValue) as total_buy_value,
                       SUM(sellValue) as total_sell_value,
                       SUM(netValue) as total_net_value
                FROM fii_dii_activity 
                WHERE TRADE_DATE = ?
                GROUP BY category
            """, [date]).df()
            
            if not fii_dii_summary.empty:
                summary['fii_dii_summary'] = fii_dii_summary.to_dict('records')
            
            logger.info(f"✅ Retrieved market summary for {date}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting market summary: {e}")
            return {}
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("✅ Closed EOD database connection")


# Global EOD data provider instance
_eod_data_provider = None


def get_eod_data_provider() -> EODDataProvider:
    """Get the global EOD data provider instance."""
    global _eod_data_provider
    if _eod_data_provider is None:
        _eod_data_provider = EODDataProvider()
    return _eod_data_provider
