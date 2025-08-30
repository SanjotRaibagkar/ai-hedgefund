#!/usr/bin/env python3
"""
Debug Next Day Outcome Calculation
Investigate why next-day returns are showing as 0.00%
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import duckdb
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

def debug_next_day_outcome():
    """Debug the next day outcome calculation issue."""
    
    print("🔍 Debugging Next Day Outcome Calculation")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Test specific cases from the output
        test_cases = [
            ('NIFTY', '2025-05-30'),
            ('RELIANCE', '2025-06-27'),
            ('BANKNIFTY', '2025-07-09'),
            ('FORTIS', '2025-06-24')
        ]
        
        for symbol, date in test_cases:
            print(f"\n📊 Testing {symbol} on {date}:")
            
            # Get current day data
            current_day_query = f"""
            SELECT 
                TckrSymb,
                TRADE_DATE,
                ClsPric,
                OpnPric,
                HghPric,
                LwPric,
                PrvsClsgPric
            FROM fno_bhav_copy
            WHERE TckrSymb = '{symbol}' 
            AND TRADE_DATE = '{date}'
            AND ClsPric > 100
            ORDER BY ClsPric DESC
            LIMIT 1
            """
            
            current_result = conn.execute(current_day_query).fetchdf()
            
            if current_result.empty:
                print(f"   ❌ No current day data found")
                continue
                
            current_data = current_result.iloc[0]
            current_close = current_data['ClsPric']
            prev_close = current_data['PrvsClsgPric']
            
            # Calculate current day return
            current_return = ((current_close - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            print(f"   Current day close: {current_close:.2f}")
            print(f"   Previous close: {prev_close:.2f}")
            print(f"   Current day return: {current_return:+.2f}%")
            
            # Get next day data
            next_date = (pd.to_datetime(date) + timedelta(days=1)).strftime('%Y-%m-%d')
            
            next_day_query = f"""
            SELECT 
                TckrSymb,
                TRADE_DATE,
                ClsPric,
                OpnPric,
                HghPric,
                LwPric
            FROM fno_bhav_copy
            WHERE TckrSymb = '{symbol}' 
            AND TRADE_DATE = '{next_date}'
            AND ClsPric > 100
            ORDER BY ClsPric DESC
            LIMIT 1
            """
            
            next_result = conn.execute(next_day_query).fetchdf()
            
            if next_result.empty:
                print(f"   ❌ No next day data found for {next_date}")
                print(f"   Next day return: 0.00% (no data)")
                continue
                
            next_data = next_result.iloc[0]
            next_close = next_data['ClsPric']
            next_high = next_data['HghPric']
            next_low = next_data['LwPric']
            
            # Calculate next day return
            next_return = ((next_close - current_close) / current_close * 100) if current_close > 0 else 0
            
            print(f"   Next day ({next_date}):")
            print(f"     Close: {next_close:.2f}")
            print(f"     High: {next_high:.2f}")
            print(f"     Low: {next_low:.2f}")
            print(f"   Next day return: {next_return:+.2f}%")
            
            # Check if this matches the vector store output
            if abs(next_return) < 0.01:
                print(f"   ⚠️ WARNING: Next day return is close to 0.00%")
            else:
                print(f"   ✅ Next day return is realistic: {next_return:+.2f}%")
        
        # Check for patterns in the data
        print(f"\n🔍 Checking for patterns in next-day data:")
        
        # Check how many next-day records exist
        pattern_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN ClsPric > 100 THEN 1 END) as valid_price_records,
            COUNT(CASE WHEN ClsPric <= 100 THEN 1 END) as low_price_records
        FROM fno_bhav_copy
        WHERE TRADE_DATE >= '2025-05-01'
        AND TRADE_DATE <= '2025-08-30'
        """
        
        pattern_result = conn.execute(pattern_query).fetchdf()
        print(f"   Total records: {pattern_result['total_records'].iloc[0]}")
        print(f"   Valid price records (>100): {pattern_result['valid_price_records'].iloc[0]}")
        print(f"   Low price records (≤100): {pattern_result['low_price_records'].iloc[0]}")
        
        # Check specific symbols for next-day availability
        symbols = ['NIFTY', 'RELIANCE', 'BANKNIFTY', 'FORTIS']
        
        for symbol in symbols:
            symbol_query = f"""
            SELECT 
                COUNT(DISTINCT TRADE_DATE) as trading_days,
                COUNT(DISTINCT CASE WHEN ClsPric > 100 THEN TRADE_DATE END) as valid_days
            FROM fno_bhav_copy
            WHERE TckrSymb = '{symbol}'
            AND TRADE_DATE >= '2025-05-01'
            AND TRADE_DATE <= '2025-08-30'
            """
            
            symbol_result = conn.execute(symbol_query).fetchdf()
            total_days = symbol_result['trading_days'].iloc[0]
            valid_days = symbol_result['valid_days'].iloc[0]
            
            print(f"   {symbol}: {valid_days}/{total_days} days have valid prices")
        
        # Check for consecutive days
        print(f"\n🔍 Checking consecutive day availability:")
        
        for symbol in symbols:
            consecutive_query = f"""
            WITH daily_data AS (
                SELECT 
                    TRADE_DATE,
                    ClsPric,
                    ROW_NUMBER() OVER (ORDER BY TRADE_DATE) as rn
                FROM fno_bhav_copy
                WHERE TckrSymb = '{symbol}'
                AND ClsPric > 100
                AND TRADE_DATE >= '2025-05-01'
                AND TRADE_DATE <= '2025-08-30'
            ),
            consecutive_check AS (
                SELECT 
                    TRADE_DATE,
                    ClsPric,
                    LAG(TRADE_DATE) OVER (ORDER BY TRADE_DATE) as prev_date,
                    LAG(ClsPric) OVER (ORDER BY TRADE_DATE) as prev_close
                FROM daily_data
            )
            SELECT 
                COUNT(*) as total_days,
                COUNT(CASE WHEN prev_date IS NOT NULL THEN 1 END) as days_with_prev,
                COUNT(CASE WHEN prev_close > 0 THEN 1 END) as days_with_valid_prev
            FROM consecutive_check
            """
            
            consecutive_result = conn.execute(consecutive_query).fetchdf()
            total_days = consecutive_result['total_days'].iloc[0]
            days_with_prev = consecutive_result['days_with_prev'].iloc[0]
            days_with_valid_prev = consecutive_result['days_with_valid_prev'].iloc[0]
            
            print(f"   {symbol}: {days_with_valid_prev}/{total_days} days have valid previous day data")
        
        conn.close()
        
        print(f"\n🔧 Root Cause Analysis:")
        print("The issue is likely that:")
        print("1. Next-day data is missing for many dates")
        print("2. The _get_next_day_outcome function returns 0.00% when no data is found")
        print("3. We need to improve the fallback logic")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_next_day_outcome()
