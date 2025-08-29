#!/usr/bin/env python3
"""
Options Chain Database Manager
Manages a separate DuckDB database for options chain data to avoid locking issues.
"""

import os
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
from loguru import logger


class OptionsDatabaseManager:
    """Manages DuckDB database operations specifically for options chain data."""
    
    def __init__(self, db_path: str = "data/options_chain_data.duckdb"):
        """
        Initialize Options DuckDB database manager.
        
        Args:
            db_path: Path to DuckDB database file for options data
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
        """Initialize database connection and create options table."""
        try:
            self.connection = duckdb.connect(self.db_path)
            self._create_options_table()
            logger.info(f"Options DuckDB database initialized successfully at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Options DuckDB database: {e}")
            raise
            
    def _create_options_table(self):
        """Create options_chain_data table if it doesn't exist."""
        
        # Options Chain Data Table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS options_chain_data (
                timestamp TIMESTAMP,
                index_symbol VARCHAR,
                strike_price DOUBLE,
                expiry_date DATE,
                option_type VARCHAR,
                last_price DOUBLE,
                bid_price DOUBLE,
                ask_price DOUBLE,
                volume BIGINT,
                open_interest BIGINT,
                change_in_oi BIGINT,
                implied_volatility DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                spot_price DOUBLE,
                atm_strike DOUBLE,
                pcr DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, index_symbol, strike_price, expiry_date, option_type)
            )
        """)
        
        logger.info("✅ Options chain data table created/verified")
        
        # Create indexes for better performance
        self._create_indexes()
        
    def _create_indexes(self):
        """Create database indexes for better query performance."""
        
        # Options data indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_options_timestamp ON options_chain_data(timestamp)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_options_index ON options_chain_data(index_symbol)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_options_strike ON options_chain_data(strike_price)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_options_expiry ON options_chain_data(expiry_date)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_options_type ON options_chain_data(option_type)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_options_index_timestamp ON options_chain_data(index_symbol, timestamp)")
        
        logger.info("✅ Options database indexes created")
    
    def insert_options_data(self, data: pd.DataFrame):
        """Insert options chain data into database."""
        try:
            if data.empty:
                logger.warning("No options data to insert")
                return
                
            # Ensure timestamp column exists
            if 'timestamp' not in data.columns:
                data['timestamp'] = datetime.now()
                
            # Ensure created_at column exists
            if 'created_at' not in data.columns:
                data['created_at'] = datetime.now()
            
            # Insert data using DuckDB's efficient DataFrame insertion
            # Register DataFrame as temporary table and insert with REPLACE to handle duplicates
            self.connection.register("temp_data", data)
            self.connection.execute("INSERT OR REPLACE INTO options_chain_data SELECT * FROM temp_data")
            self.connection.execute("DROP VIEW temp_data")
            
            logger.info(f"Inserted {len(data)} options data records")
            
        except Exception as e:
            logger.error(f"Failed to insert options data: {e}")
            raise
    
    def get_recent_options_data(self, index_symbol: str, minutes: int = 60) -> pd.DataFrame:
        """Get recent options data for a specific index."""
        try:
            query = """
                SELECT * FROM options_chain_data 
                WHERE index_symbol = ? 
                AND timestamp >= NOW() - INTERVAL '{} minutes'
                ORDER BY timestamp DESC
            """.format(minutes)
            
            result = self.connection.execute(query, [index_symbol]).fetchdf()
            logger.info(f"Retrieved {len(result)} recent options records for {index_symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get recent options data: {e}")
            return pd.DataFrame()
    
    def get_latest_spot_price(self, index_symbol: str) -> Optional[float]:
        """Get the latest spot price for a specific index."""
        try:
            query = """
                SELECT spot_price 
                FROM options_chain_data 
                WHERE index_symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            result = self.connection.execute(query, [index_symbol]).fetchone()
            if result:
                return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest spot price: {e}")
            return None
    
    def get_daily_summary(self, index_symbol: str, date: str = None) -> Dict[str, Any]:
        """Get daily summary for options data."""
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
                
            query = """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record,
                    AVG(spot_price) as avg_spot_price,
                    MIN(spot_price) as min_spot_price,
                    MAX(spot_price) as max_spot_price
                FROM options_chain_data 
                WHERE index_symbol = ? 
                AND DATE(timestamp) = ?
            """
            
            result = self.connection.execute(query, [index_symbol, date]).fetchone()
            
            if result:
                return {
                    'index_symbol': index_symbol,
                    'date': date,
                    'total_records': result[0],
                    'first_record': result[1],
                    'last_record': result[2],
                    'avg_spot_price': result[3],
                    'min_spot_price': result[4],
                    'max_spot_price': result[5]
                }
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get daily summary: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Options database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
