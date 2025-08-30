#!/usr/bin/env python3
"""
Debug Instrument Types
Investigate the F&O instrument types and data structure
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import duckdb
import pandas as pd
from loguru import logger

def debug_instrument_types():
    """Debug the instrument types and data structure."""
    
    print("🔍 Debugging Instrument Types")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Check all instrument types
        print("1. All instrument types in F&O data:")
        result = conn.execute("""
            SELECT DISTINCT FinInstrmTp, COUNT(*) as count
            FROM fno_bhav_copy
            GROUP BY FinInstrmTp
            ORDER BY FinInstrmTp
        """).fetchdf()
        
        print(result)
        
        # Check BANKNIFTY data structure
        print("\n2. BANKNIFTY data structure (2025-08-01):")
        banknifty_data = conn.execute("""
            SELECT 
                FinInstrmTp,
                COUNT(*) as count,
                MIN(ClsPric) as min_strike,
                MAX(ClsPric) as max_strike,
                MIN(OpnIntrst) as min_oi,
                MAX(OpnIntrst) as max_oi,
                SUM(OpnIntrst) as total_oi
            FROM fno_bhav_copy
            WHERE TckrSymb = 'BANKNIFTY' 
            AND TRADE_DATE = '2025-08-01'
            GROUP BY FinInstrmTp
            ORDER BY FinInstrmTp
        """).fetchdf()
        
        print(banknifty_data)
        
        # Check sample data for each instrument type
        print("\n3. Sample data for each instrument type:")
        for inst_type in ['IDF', 'STF', 'IDO', 'STO']:
            print(f"\n   {inst_type}:")
            sample = conn.execute(f"""
                SELECT 
                    TckrSymb,
                    FinInstrmTp,
                    OpnPric,
                    ClsPric,
                    OpnIntrst,
                    ChngInOpnIntrst,
                    TtlTradgVol
                FROM fno_bhav_copy
                WHERE TckrSymb = 'BANKNIFTY' 
                AND TRADE_DATE = '2025-08-01'
                AND FinInstrmTp = '{inst_type}'
                AND OpnIntrst > 0
                ORDER BY OpnIntrst DESC
                LIMIT 5
            """).fetchdf()
            
            print(sample)
        
        # Check if there are any calls vs puts
        print("\n4. Checking for calls vs puts:")
        
        # Look for patterns in strike prices
        strike_analysis = conn.execute("""
            SELECT 
                FinInstrmTp,
                COUNT(*) as count,
                AVG(ClsPric) as avg_strike,
                MIN(ClsPric) as min_strike,
                MAX(ClsPric) as max_strike,
                SUM(OpnIntrst) as total_oi
            FROM fno_bhav_copy
            WHERE TckrSymb = 'BANKNIFTY' 
            AND TRADE_DATE = '2025-08-01'
            AND FinInstrmTp IN ('IDO', 'STO')
            GROUP BY FinInstrmTp
        """).fetchdf()
        
        print("Strike price analysis:")
        print(strike_analysis)
        
        # Check if STO might be puts and IDO might be calls
        print("\n5. Testing alternative mapping (STO=puts, IDO=calls):")
        
        # Get stock price
        stock_price = conn.execute("""
            SELECT ClsPric
            FROM fno_bhav_copy
            WHERE TckrSymb = 'BANKNIFTY' 
            AND TRADE_DATE = '2025-08-01'
            AND ClsPric > 100
            ORDER BY ClsPric DESC
            LIMIT 1
        """).fetchdf()
        
        if not stock_price.empty:
            spot_price = stock_price['ClsPric'].iloc[0]
            print(f"   Spot price: {spot_price}")
            
            # Test with STO as puts, IDO as calls
            puts_sto = conn.execute(f"""
                SELECT SUM(OpnIntrst) as put_oi
                FROM fno_bhav_copy
                WHERE TckrSymb = 'BANKNIFTY' 
                AND TRADE_DATE = '2025-08-01'
                AND FinInstrmTp = 'STO'
                AND OpnIntrst > 0
            """).fetchdf()
            
            calls_ido = conn.execute(f"""
                SELECT SUM(OpnIntrst) as call_oi
                FROM fno_bhav_copy
                WHERE TckrSymb = 'BANKNIFTY' 
                AND TRADE_DATE = '2025-08-01'
                AND FinInstrmTp = 'IDO'
                AND OpnIntrst > 0
            """).fetchdf()
            
            put_oi = puts_sto['put_oi'].iloc[0] if not puts_sto.empty else 0
            call_oi = calls_ido['call_oi'].iloc[0] if not calls_ido.empty else 0
            
            print(f"   STO (puts?): {put_oi}")
            print(f"   IDO (calls?): {call_oi}")
            
            if call_oi > 0:
                pcr_swapped = put_oi / call_oi
                print(f"   PCR (swapped): {pcr_swapped:.3f}")
            else:
                print(f"   PCR (swapped): N/A (no call OI)")
        
        # Check NIFTY data for comparison
        print("\n6. Checking NIFTY data for comparison:")
        nifty_data = conn.execute("""
            SELECT 
                FinInstrmTp,
                COUNT(*) as count,
                SUM(OpnIntrst) as total_oi
            FROM fno_bhav_copy
            WHERE TckrSymb = 'NIFTY' 
            AND TRADE_DATE = '2025-08-01'
            GROUP BY FinInstrmTp
            ORDER BY FinInstrmTp
        """).fetchdf()
        
        print("NIFTY data:")
        print(nifty_data)
        
        conn.close()
        
        print("\n🔧 Analysis:")
        print("Based on the data, it appears that:")
        print("1. STO has much more data than IDO")
        print("2. STO might be the main options type (calls or puts)")
        print("3. IDO might be a different instrument type")
        print("4. We need to verify the correct mapping")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_instrument_types()
