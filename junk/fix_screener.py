#!/usr/bin/env python3
"""
Quick fix for DuckDB screener to use only database data
"""

import duckdb
import pandas as pd
from datetime import datetime
from loguru import logger

def get_latest_price_from_duckdb(symbol: str, db_path: str = "data/comprehensive_equity.duckdb"):
    """Get latest price data from DuckDB instead of live API."""
    try:
        with duckdb.connect(db_path) as conn:
            query = """
                SELECT symbol, date, open_price, high_price, low_price, close_price, volume, turnover
                FROM price_data 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            """
            
            result = conn.execute(query, [symbol]).fetchdf()
            
            if not result.empty:
                row = result.iloc[0]
                return {
                    'LastTradedPrice': row['close_price'],
                    'Open': row['open_price'],
                    'High': row['high_price'],
                    'Low': row['low_price'],
                    'Volume': row['volume'],
                    'Date': row['date']
                }
            return None
            
    except Exception as e:
        logger.error(f"Error getting latest price for {symbol}: {e}")
        return None

# Test the function
if __name__ == "__main__":
    test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
    for symbol in test_symbols:
        price_data = get_latest_price_from_duckdb(symbol)
        if price_data:
            print(f"{symbol}: ₹{price_data['LastTradedPrice']} (Volume: {price_data['Volume']:,})")
        else:
            print(f"{symbol}: No data found")
