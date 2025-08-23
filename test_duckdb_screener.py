#!/usr/bin/env python3
"""Test DuckDB Screener"""

import duckdb
import pandas as pd
from src.nsedata.NseUtility import NseUtils

def test_duckdb_data():
    """Test DuckDB data access."""
    print("Testing DuckDB data access...")
    
    conn = duckdb.connect('data/comprehensive_equity.duckdb')
    
    # Check tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Tables: {tables}")
    
    # Get sample data
    acc_data = conn.execute("SELECT * FROM price_data WHERE symbol = 'ACC' ORDER BY date DESC LIMIT 5").fetchall()
    print(f"ACC Historical Data (first 5 rows):")
    for row in acc_data:
        print(f"  {row}")
    
    # Check data structure
    schema = conn.execute("DESCRIBE price_data").fetchall()
    print(f"Price data schema:")
    for col in schema:
        print(f"  {col}")
    
    conn.close()

def test_nse_utility():
    """Test NSEUtility data."""
    print("\nTesting NSEUtility...")
    
    nse = NseUtils()
    acc_data = nse.price_info('ACC')
    print(f"ACC Current Data: {acc_data}")
    
    # Check if volume is available
    if 'Volume' in acc_data:
        print(f"Volume: {acc_data['Volume']}")
    else:
        print("No volume data in NSEUtility response")

def test_screener_logic():
    """Test screener logic."""
    print("\nTesting screener logic...")
    
    # Simulate the screening process
    from src.nsedata.NseUtility import NseUtils
    nse = NseUtils()
    
    symbol = 'ACC'
    min_volume = 1000
    min_price = 1.0
    
    # Get current data
    price_info = nse.price_info(symbol)
    print(f"Price info: {price_info}")
    
    if not price_info:
        print("No price info available")
        return
    
    current_price = float(price_info.get('LastTradedPrice', 0))
    volume = int(price_info.get('Volume', 0))
    
    print(f"Current price: {current_price}")
    print(f"Volume: {volume}")
    print(f"Min volume filter: {min_volume}")
    print(f"Min price filter: {min_price}")
    
    # Check filters
    if volume < min_volume:
        print(f"❌ Volume {volume} < {min_volume}")
    else:
        print(f"✅ Volume {volume} >= {min_volume}")
    
    if current_price < min_price:
        print(f"❌ Price {current_price} < {min_price}")
    else:
        print(f"✅ Price {current_price} >= {min_price}")

if __name__ == "__main__":
    test_duckdb_data()
    test_nse_utility()
    test_screener_logic() 