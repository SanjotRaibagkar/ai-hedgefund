"""
SQLite Database Manager for AI Hedge Fund Data Storage
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
from loguru import logger

from .models import (
    TechnicalData,
    FundamentalData,
    MarketData,
    CorporateActions,
    DataQualityMetrics
)

class DatabaseManager:
    """Manages SQLite database operations for the AI Hedge Fund."""
    
    def __init__(self, db_path: str = "data/ai_hedge_fund.db"):
        """
        Initialize SQLite database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self._ensure_data_directory()
        self._initialize_database()
        
    def _ensure_data_directory(self):
        """Ensure data directory exists."""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            self._create_tables()
            logger.info(f"SQLite database initialized successfully at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
            
    def _create_tables(self):
        """Create database tables if they don't exist."""
        
        # Technical Data Table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS technical_data (
                ticker TEXT,
                trade_date TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                adjusted_close REAL,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                bollinger_upper REAL,
                bollinger_lower REAL,
                bollinger_middle REAL,
                atr_14 REAL,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ticker, trade_date)
            )
        """)
        
        # Fundamental Data Table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_data (
                ticker TEXT,
                report_date TEXT,
                period_type TEXT,
                revenue REAL,
                net_income REAL,
                total_assets REAL,
                total_liabilities REAL,
                total_equity REAL,
                operating_cash_flow REAL,
                free_cash_flow REAL,
                debt_to_equity REAL,
                roe REAL,
                roa REAL,
                pe_ratio REAL,
                pb_ratio REAL,
                ps_ratio REAL,
                dividend_yield REAL,
                market_cap REAL,
                enterprise_value REAL,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ticker, report_date, period_type)
            )
        """)
        
        # Market Data Table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                ticker TEXT,
                market_date TEXT,
                market_cap REAL,
                enterprise_value REAL,
                beta REAL,
                dividend_yield REAL,
                payout_ratio REAL,
                price_to_book REAL,
                price_to_sales REAL,
                price_to_earnings REAL,
                forward_pe REAL,
                peg_ratio REAL,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ticker, market_date)
            )
        """)
        
        # Corporate Actions Table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                ticker TEXT,
                action_date TEXT,
                action_type TEXT,
                description TEXT,
                value REAL,
                ratio TEXT,
                ex_date TEXT,
                record_date TEXT,
                payment_date TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (ticker, action_date, action_type)
            )
        """)
        
        # Data Quality Metrics Table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                ticker TEXT,
                quality_date TEXT,
                data_type TEXT,
                completeness_score REAL,
                accuracy_score REAL,
                timeliness_score REAL,
                consistency_score REAL,
                total_records INTEGER,
                missing_records INTEGER,
                error_count INTEGER,
                last_updated TEXT,
                created_at TEXT,
                PRIMARY KEY (ticker, quality_date, data_type)
            )
        """)
        
        # Create indexes for better performance
        self._create_indexes()
        self.connection.commit()
        
    def _create_indexes(self):
        """Create database indexes for better query performance."""
        
        # Technical data indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_technical_ticker ON technical_data(ticker)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_technical_date ON technical_data(trade_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_technical_ticker_date ON technical_data(ticker, trade_date)")
        
        # Fundamental data indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_ticker ON fundamental_data(ticker)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_date ON fundamental_data(report_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_period ON fundamental_data(period_type)")
        
        # Market data indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_market_ticker ON market_data(ticker)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(market_date)")
        
        # Corporate actions indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_corporate_ticker ON corporate_actions(ticker)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_corporate_date ON corporate_actions(action_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_corporate_type ON corporate_actions(action_type)")
        
    def insert_technical_data(self, data: Union[pd.DataFrame, List[TechnicalData]]):
        """Insert technical data into database."""
        try:
            if isinstance(data, list):
                # Convert TechnicalData objects to DataFrame
                df = pd.DataFrame([item.dict() for item in data])
            else:
                df = data.copy()
            
            # Ensure column names match database schema
            column_mapping = {
                'date': 'trade_date',
                'open': 'open_price',
                'high': 'high_price',
                'low': 'low_price',
                'close': 'close_price'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert date columns to string format
            if 'trade_date' in df.columns:
                df['trade_date'] = df['trade_date'].astype(str)
            
            # Insert data
            df.to_sql('technical_data', self.connection, if_exists='append', 
                     method='multi', index=False)
            self.connection.commit()
            
            logger.info(f"Inserted {len(df)} technical data records")
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to insert technical data: {e}")
            raise
    
    def insert_fundamental_data(self, data: Union[pd.DataFrame, List[FundamentalData]]):
        """Insert fundamental data into database."""
        try:
            if isinstance(data, list):
                # Convert FundamentalData objects to DataFrame
                df = pd.DataFrame([item.dict() for item in data])
            else:
                df = data.copy()
            
            # Ensure column names match database schema
            column_mapping = {
                'date': 'report_date'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert date columns to string format
            if 'report_date' in df.columns:
                df['report_date'] = df['report_date'].astype(str)
            
            # Insert data
            df.to_sql('fundamental_data', self.connection, if_exists='append', 
                     method='multi', index=False)
            self.connection.commit()
            
            logger.info(f"Inserted {len(df)} fundamental data records")
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to insert fundamental data: {e}")
            raise
    
    def get_technical_data(self, ticker: str, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve technical data from database."""
        try:
            query = "SELECT * FROM technical_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND trade_date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND trade_date <= ?"
                params.append(end_date)
            
            query += " ORDER BY trade_date"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            # Convert date columns back to datetime
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            logger.info(f"Retrieved {len(df)} technical data records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve technical data: {e}")
            raise
    
    def get_fundamental_data(self, ticker: str, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None, 
                            period_type: Optional[str] = None) -> pd.DataFrame:
        """Retrieve fundamental data from database."""
        try:
            query = "SELECT * FROM fundamental_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND report_date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND report_date <= ?"
                params.append(end_date)
            
            if period_type:
                query += " AND period_type = ?"
                params.append(period_type)
            
            query += " ORDER BY report_date"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            
            # Convert date columns back to datetime
            if 'report_date' in df.columns:
                df['report_date'] = pd.to_datetime(df['report_date'])
            
            logger.info(f"Retrieved {len(df)} fundamental data records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve fundamental data: {e}")
            raise
    
    def get_latest_data_date(self, ticker: str, data_type: str = 'technical') -> Optional[str]:
        """Get the latest data date for a ticker."""
        try:
            if data_type == 'technical':
                query = "SELECT MAX(trade_date) as latest_date FROM technical_data WHERE ticker = ?"
            elif data_type == 'fundamental':
                query = "SELECT MAX(report_date) as latest_date FROM fundamental_data WHERE ticker = ?"
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            cursor = self.connection.cursor()
            cursor.execute(query, (ticker,))
            result = cursor.fetchone()
            
            return result[0] if result and result[0] else None
            
        except Exception as e:
            logger.error(f"Failed to get latest data date: {e}")
            raise
    
    def get_data_quality_metrics(self, ticker: str, data_type: str = 'technical') -> pd.DataFrame:
        """Get data quality metrics for a ticker."""
        try:
            query = """
                SELECT * FROM data_quality_metrics 
                WHERE ticker = ? AND data_type = ?
                ORDER BY quality_date DESC
            """
            
            df = pd.read_sql_query(query, self.connection, params=[ticker, data_type])
            
            # Convert date columns back to datetime
            if 'quality_date' in df.columns:
                df['quality_date'] = pd.to_datetime(df['quality_date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get data quality metrics: {e}")
            raise
    
    def get_missing_data_dates(self, ticker: str, start_date: str, end_date: str, 
                              data_type: str = 'technical') -> List[str]:
        """Get missing data dates for a ticker within a date range."""
        try:
            # Get all trading days in the range (excluding weekends)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            trading_days = [d.strftime('%Y-%m-%d') for d in date_range 
                          if d.weekday() < 5]  # Monday = 0, Friday = 4
            
            # Get existing data dates
            if data_type == 'technical':
                query = """
                    SELECT trade_date FROM technical_data 
                    WHERE ticker = ? AND trade_date BETWEEN ? AND ?
                """
            elif data_type == 'fundamental':
                query = """
                    SELECT report_date FROM fundamental_data 
                    WHERE ticker = ? AND report_date BETWEEN ? AND ?
                """
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            cursor = self.connection.cursor()
            cursor.execute(query, (ticker, start_date, end_date))
            existing_dates = [row[0] for row in cursor.fetchall()]
            
            # Find missing dates
            missing_dates = [date for date in trading_days if date not in existing_dates]
            
            return missing_dates
            
        except Exception as e:
            logger.error(f"Failed to get missing data dates: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

# Alias for backward compatibility
DuckDBManager = DatabaseManager 