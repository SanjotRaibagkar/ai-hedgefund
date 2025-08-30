#!/usr/bin/env python3
"""
Debug PCR Calculation
Investigate why PCR is always 1.00
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import duckdb
import pandas as pd
from loguru import logger

def debug_pcr_calculation():
    """Debug the PCR calculation issue."""
    
    print("🔍 Debugging PCR Calculation")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Check F&O data structure
        print("1. Checking F&O data structure:")
        result = conn.execute("""
            SELECT DISTINCT FinInstrmTp, COUNT(*) as count
            FROM fno_bhav_copy
            WHERE FinInstrmTp IN ('IDF', 'STF', 'IDO', 'STO')
            GROUP BY FinInstrmTp
            ORDER BY FinInstrmTp
        """).fetchdf()
        
        print("Instrument types in F&O data:")
        print(result)
        
        # Check specific symbol data
        print("\n2. Checking BANKNIFTY options data:")
        banknifty_data = conn.execute("""
            SELECT 
                TckrSymb,
                TRADE_DATE,
                FinInstrmTp,
                OpnPric,
                ClsPric,
                OpnIntrst,
                ChngInOpnIntrst,
                TtlTradgVol
            FROM fno_bhav_copy
            WHERE TckrSymb = 'BANKNIFTY' 
            AND TRADE_DATE = '2025-08-01'
            AND FinInstrmTp IN ('IDO', 'STO')
            ORDER BY FinInstrmTp, OpnPric
            LIMIT 20
        """).fetchdf()
        
        print("BANKNIFTY options data (2025-08-01):")
        print(banknifty_data)
        
        # Check stock price for reference
        print("\n3. Checking BANKNIFTY stock price:")
        stock_price = conn.execute("""
            SELECT 
                TckrSymb,
                TRADE_DATE,
                ClsPric,
                OpnPric,
                HghPric,
                LwPric
            FROM fno_bhav_copy
            WHERE TckrSymb = 'BANKNIFTY' 
            AND TRADE_DATE = '2025-08-01'
            AND ClsPric > 100
            ORDER BY ClsPric DESC
            LIMIT 1
        """).fetchdf()
        
        print("BANKNIFTY stock price:")
        print(stock_price)
        
        if not stock_price.empty:
            spot_price = stock_price['ClsPric'].iloc[0]
            print(f"Spot price: {spot_price}")
            
            # Check options around ATM
            atm_range = spot_price * 0.02
            atm_options = conn.execute(f"""
                SELECT 
                    FinInstrmTp,
                    OpnPric,
                    OpnIntrst,
                    ChngInOpnIntrst
                FROM fno_bhav_copy
                WHERE TckrSymb = 'BANKNIFTY' 
                AND TRADE_DATE = '2025-08-01'
                AND FinInstrmTp IN ('IDO', 'STO')
                AND OpnPric >= {spot_price - atm_range}
                AND OpnPric <= {spot_price + atm_range}
                ORDER BY FinInstrmTp, OpnPric
            """).fetchdf()
            
            print(f"\n4. Options around ATM (±{atm_range:.2f}):")
            print(atm_options)
            
            # Calculate PCR manually
            puts = atm_options[atm_options['FinInstrmTp'] == 'IDO']
            calls = atm_options[atm_options['FinInstrmTp'] == 'STO']
            
            put_oi = puts['OpnIntrst'].sum() if not puts.empty else 0
            call_oi = calls['OpnIntrst'].sum() if not calls.empty else 0
            
            print(f"\n5. Manual PCR Calculation:")
            print(f"   Put OI: {put_oi}")
            print(f"   Call OI: {call_oi}")
            print(f"   PCR: {put_oi / call_oi if call_oi > 0 else 'N/A'}")
            
            # Check if we need to swap IDO/STO
            print(f"\n6. Checking if IDO/STO mapping is correct:")
            print(f"   IDO (puts?): {len(puts)} records, OI: {put_oi}")
            print(f"   STO (calls?): {len(calls)} records, OI: {call_oi}")
            
            # Try swapping
            puts_swapped = atm_options[atm_options['FinInstrmTp'] == 'STO']
            calls_swapped = atm_options[atm_options['FinInstrmTp'] == 'IDO']
            
            put_oi_swapped = puts_swapped['OpnIntrst'].sum() if not puts_swapped.empty else 0
            call_oi_swapped = calls_swapped['OpnIntrst'].sum() if not calls_swapped.empty else 0
            
            print(f"\n7. Swapped PCR Calculation:")
            print(f"   Put OI (STO): {put_oi_swapped}")
            print(f"   Call OI (IDO): {call_oi_swapped}")
            print(f"   PCR: {put_oi_swapped / call_oi_swapped if call_oi_swapped > 0 else 'N/A'}")
        
        # Check multiple dates
        print("\n8. Checking PCR across multiple dates:")
        dates = ['2025-08-01', '2025-08-06', '2025-03-17']
        
        for date in dates:
            print(f"\n   Date: {date}")
            
            # Get stock price
            stock_result = conn.execute(f"""
                SELECT ClsPric
                FROM fno_bhav_copy
                WHERE TckrSymb = 'BANKNIFTY' 
                AND TRADE_DATE = '{date}'
                AND ClsPric > 100
                ORDER BY ClsPric DESC
                LIMIT 1
            """).fetchdf()
            
            if not stock_result.empty:
                spot = stock_result['ClsPric'].iloc[0]
                atm_range = spot * 0.02
                
                # Get options data
                options_result = conn.execute(f"""
                    SELECT 
                        FinInstrmTp,
                        SUM(OpnIntrst) as total_oi
                    FROM fno_bhav_copy
                    WHERE TckrSymb = 'BANKNIFTY' 
                    AND TRADE_DATE = '{date}'
                    AND FinInstrmTp IN ('IDO', 'STO')
                    AND OpnPric >= {spot - atm_range}
                    AND OpnPric <= {spot + atm_range}
                    GROUP BY FinInstrmTp
                """).fetchdf()
                
                print(f"      Spot: {spot:.2f}, Range: ±{atm_range:.2f}")
                print(f"      Options: {options_result.to_dict('records')}")
        
        conn.close()
        
        print("\n🔧 Root Cause Analysis:")
        print("The issue is likely in the instrument type mapping (IDO/STO)")
        print("or the ATM range calculation is too narrow")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_pcr_calculation()
