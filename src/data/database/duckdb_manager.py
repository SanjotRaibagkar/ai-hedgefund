"""
DuckDB Database Manager for AI Hedge Fund Data Storage
"""

import os
import duckdb
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
    """Manages DuckDB database operations for the AI Hedge Fund."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        """
        Initialize DuckDB database manager.
        
        Args:
            db_path: Path to DuckDB database file
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
            self.connection = duckdb.connect(self.db_path)
            
            # DuckDB has built-in concurrency support
            # No need for SQLite PRAGMA commands
            
            self._create_tables()
            logger.info(f"DuckDB database initialized successfully at {self.db_path} with built-in concurrency support")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB database: {e}")
            raise
            
    def _create_tables(self):
        """Create database tables if they don't exist."""
        
        # FIXED: Use CREATE TABLE IF NOT EXISTS instead of DROP TABLE
        # This preserves existing data and only creates tables if they don't exist
        logger.info("🔧 Creating tables if they don't exist (preserving existing data)...")
        
        # Technical Data Table with foreign key to price_data
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS technical_data (
                ticker VARCHAR,
                trade_date DATE,
                open_price DOUBLE,
                high_price DOUBLE,
                low_price DOUBLE,
                close_price DOUBLE,
                volume BIGINT,
                adjusted_close DOUBLE,
                sma_20 DOUBLE,
                sma_50 DOUBLE,
                sma_200 DOUBLE,
                rsi_14 DOUBLE,
                macd DOUBLE,
                macd_signal DOUBLE,
                macd_histogram DOUBLE,
                bollinger_upper DOUBLE,
                bollinger_lower DOUBLE,
                bollinger_middle DOUBLE,
                atr_14 DOUBLE,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (ticker, trade_date),
                FOREIGN KEY (ticker, trade_date) REFERENCES price_data(symbol, date)
            )
        """)
        
        # Fundamental Data Table with foreign key to price_data
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_data (
                ticker VARCHAR,
                report_date DATE,
                period_type VARCHAR,
                revenue DOUBLE,
                net_income DOUBLE,
                total_assets DOUBLE,
                total_liabilities DOUBLE,
                total_equity DOUBLE,
                operating_cash_flow DOUBLE,
                free_cash_flow DOUBLE,
                debt_to_equity DOUBLE,
                roe DOUBLE,
                roa DOUBLE,
                pe_ratio DOUBLE,
                pb_ratio DOUBLE,
                ps_ratio DOUBLE,
                dividend_yield DOUBLE,
                market_cap DOUBLE,
                enterprise_value DOUBLE,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (ticker, report_date, period_type),
                FOREIGN KEY (ticker, report_date) REFERENCES price_data(symbol, date)
            )
        """)
        
        # Market Data Table with foreign key to price_data
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                ticker VARCHAR,
                market_date DATE,
                market_cap DOUBLE,
                enterprise_value DOUBLE,
                beta DOUBLE,
                dividend_yield DOUBLE,
                payout_ratio DOUBLE,
                price_to_book DOUBLE,
                price_to_sales DOUBLE,
                price_to_earnings DOUBLE,
                forward_pe DOUBLE,
                peg_ratio DOUBLE,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (ticker, market_date),
                FOREIGN KEY (ticker, market_date) REFERENCES price_data(symbol, date)
            )
        """)
        
        # Corporate Actions Table with foreign key to price_data
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                ticker VARCHAR,
                action_date DATE,
                action_type VARCHAR,
                description TEXT,
                ex_date DATE,
                record_date DATE,
                payment_date DATE,
                ratio VARCHAR,
                dividend_amount DOUBLE,
                split_ratio VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (ticker, action_date, action_type),
                FOREIGN KEY (ticker, action_date) REFERENCES price_data(symbol, date)
            )
        """)
        
        # Data Quality Metrics Table (using existing schema)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                ticker VARCHAR,
                quality_date DATE,
                data_type VARCHAR,
                completeness_score DOUBLE,
                accuracy_score DOUBLE,
                timeliness_score DOUBLE,
                consistency_score DOUBLE,
                total_records INTEGER,
                missing_records INTEGER,
                error_count INTEGER,
                last_updated TIMESTAMP,
                created_at TIMESTAMP,
                PRIMARY KEY (ticker, quality_date, data_type)
            )
        """)
        
        # Create indexes for better performance
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_technical_ticker_date ON technical_data(ticker, trade_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_ticker_date ON fundamental_data(ticker, report_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_market_ticker_date ON market_data(ticker, market_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_corporate_ticker_date ON corporate_actions(ticker, action_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_quality_ticker_date ON data_quality_metrics(ticker, quality_date)")
        
        logger.info("✅ All tables created/verified successfully (existing data preserved)")
        
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
            
            # Ensure date columns are in proper format for DuckDB
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
            
            # Ensure timestamp columns are in proper format
            if 'created_at' not in df.columns:
                df['created_at'] = datetime.now()
            if 'updated_at' not in df.columns:
                df['updated_at'] = datetime.now()
            
            # Insert data using DuckDB's efficient DataFrame insertion
            self.connection.execute("INSERT INTO technical_data SELECT * FROM df")
            
            logger.info(f"Inserted {len(df)} technical data records")
            
        except Exception as e:
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
            
            # Ensure date columns are in proper format for DuckDB
            if 'report_date' in df.columns:
                df['report_date'] = pd.to_datetime(df['report_date']).dt.date
            
            # Ensure timestamp columns are in proper format
            if 'created_at' not in df.columns:
                df['created_at'] = datetime.now()
            if 'updated_at' not in df.columns:
                df['updated_at'] = datetime.now()
            
            # Insert data using DuckDB's efficient DataFrame insertion
            self.connection.execute("INSERT INTO fundamental_data SELECT * FROM df")
            
            logger.info(f"Inserted {len(df)} fundamental data records")
            
        except Exception as e:
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
            
            df = self.connection.execute(query, params).fetchdf()
            
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
            
            df = self.connection.execute(query, params).fetchdf()
            
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
            
            result = self.connection.execute(query, [ticker]).fetchone()
            
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
            
            df = self.connection.execute(query, [ticker, data_type]).fetchdf()
            
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
            
            result = self.connection.execute(query, [ticker, start_date, end_date]).fetchall()
            existing_dates = [row[0].strftime('%Y-%m-%d') if hasattr(row[0], 'strftime') else str(row[0]) 
                            for row in result]
            
            # Find missing dates
            missing_dates = [date for date in trading_days if date not in existing_dates]
            
            return missing_dates
            
        except Exception as e:
            logger.error(f"Failed to get missing data dates: {e}")
            raise
    
    def get_price_data(self, ticker: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve price data from the master price_data table."""
        try:
            query = "SELECT * FROM price_data WHERE symbol = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = self.connection.execute(query, params).fetchdf()
            
            logger.info(f"Retrieved {len(df)} price data records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve price data: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all available symbols in price_data table."""
        try:
            query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
            result = self.connection.execute(query).fetchall()
            symbols = [row[0] for row in result]
            
            logger.info(f"Retrieved {len(symbols)} available symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

# Alias for backward compatibility
DuckDBManager = DatabaseManager 