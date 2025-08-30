#!/usr/bin/env python3
"""
Debug Next Day Return Calculation
Investigate why we're getting unrealistic returns like -64.27% for BANKNIFTY
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import duckdb
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

def debug_next_day_calculation():
    """Debug the next day return calculation issue."""
    
    print("🔍 Debugging Next Day Return Calculation")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Check BANKNIFTY data
        print("\n1. Checking BANKNIFTY data structure:")
        result = conn.execute("""
            SELECT TckrSymb, TRADE_DATE, ClsPric, PrvsClsgPric, HghPric, LwPric
            FROM fno_bhav_copy 
            WHERE TckrSymb = 'BANKNIFTY' 
            ORDER BY TRADE_DATE DESC 
            LIMIT 10
        """).fetchdf()
        
        print(result)
        
        # Check specific problematic dates
        print("\n2. Checking specific dates from the error:")
        problem_dates = ['2025-03-17', '2025-08-20', '2025-03-19']
        
        for date in problem_dates:
            print(f"\n📅 Date: {date}")
            
            # Current day data
            current_day = conn.execute(f"""
                SELECT TckrSymb, TRADE_DATE, ClsPric, PrvsClsgPric, HghPric, LwPric
                FROM fno_bhav_copy 
                WHERE TckrSymb = 'BANKNIFTY' AND TRADE_DATE = '{date}'
                LIMIT 1
            """).fetchdf()
            
            print(f"Current day data:")
            print(current_day)
            
            # Next day data
            next_date = (pd.to_datetime(date) + timedelta(days=1)).strftime('%Y-%m-%d')
            next_day = conn.execute(f"""
                SELECT TckrSymb, TRADE_DATE, ClsPric, PrvsClsgPric, HghPric, LwPric
                FROM fno_bhav_copy 
                WHERE TckrSymb = 'BANKNIFTY' AND TRADE_DATE = '{next_date}'
                LIMIT 1
            """).fetchdf()
            
            print(f"Next day data ({next_date}):")
            print(next_day)
            
            if not current_day.empty and not next_day.empty:
                current_close = current_day['ClsPric'].iloc[0]
                next_close = next_day['ClsPric'].iloc[0]
                next_prev_close = next_day['PrvsClsgPric'].iloc[0]
                
                # Calculate returns using different methods
                correct_return = ((next_close - current_close) / current_close * 100)
                wrong_return = ((next_close - next_prev_close) / next_prev_close * 100)
                
                print(f"📊 Return Calculations:")
                print(f"   Current day close: {current_close}")
                print(f"   Next day close: {next_close}")
                print(f"   Next day prev close: {next_prev_close}")
                print(f"   CORRECT return: {correct_return:+.2f}%")
                print(f"   WRONG return (current method): {wrong_return:+.2f}%")
        
        # Check data quality issues
        print("\n3. Checking for data quality issues:")
        
        # Check for extreme values
        extreme_values = conn.execute("""
            SELECT TckrSymb, TRADE_DATE, ClsPric, PrvsClsgPric,
                   ((ClsPric - PrvsClsgPric) / PrvsClsgPric * 100) as daily_return
            FROM fno_bhav_copy 
            WHERE ABS(((ClsPric - PrvsClsgPric) / PrvsClsgPric * 100)) > 50
            ORDER BY ABS(((ClsPric - PrvsClsgPric) / PrvsClsgPric * 100)) DESC
            LIMIT 10
        """).fetchdf()
        
        print("Extreme daily returns (>50%):")
        print(extreme_values)
        
        # Check for missing or zero values
        missing_data = conn.execute("""
            SELECT TckrSymb, TRADE_DATE, ClsPric, PrvsClsgPric
            FROM fno_bhav_copy 
            WHERE ClsPric = 0 OR PrvsClsgPric = 0 OR ClsPric IS NULL OR PrvsClsgPric IS NULL
            LIMIT 10
        """).fetchdf()
        
        print("\nMissing or zero values:")
        print(missing_data)
        
        conn.close()
        
        print("\n🔧 Root Cause Analysis:")
        print("The issue is in the _get_next_day_outcome function in build_enhanced_vector_store.py")
        print("It's using PrvsClsgPric from the next day instead of ClsPric from the current day")
        print("This causes incorrect return calculations and extreme values")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_next_day_calculation()

