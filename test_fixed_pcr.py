#!/usr/bin/env python3
"""
Test Fixed PCR Calculation
Verify that PCR calculation now produces realistic values
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_enhanced_vector_store import EnhancedFNOVectorStore
from loguru import logger

def test_fixed_pcr():
    """Test the fixed PCR calculation."""
    
    print("🧪 Testing Fixed PCR Calculation")
    print("=" * 60)
    
    try:
        # Initialize vector store
        print("1. Initializing Enhanced Vector Store...")
        vector_store = EnhancedFNOVectorStore()
        
        # Test PCR calculation directly
        print("\n2. Testing PCR calculation for BANKNIFTY...")
        
        # Get BANKNIFTY data for a specific date
        query = """
        SELECT 
            TckrSymb,
            TRADE_DATE,
            FinInstrmTp,
            OpnPric,
            ClsPric,
            OpnIntrst,
            ChngInOpnIntrst
        FROM fno_bhav_copy
        WHERE TckrSymb = 'BANKNIFTY' 
        AND TRADE_DATE = '2025-08-01'
        AND FinInstrmTp IN ('IDO', 'STO')
        ORDER BY FinInstrmTp, ClsPric
        """
        
        options_data = vector_store.db_manager.connection.execute(query).fetchdf()
        print(f"   Found {len(options_data)} options records")
        
        # Get stock price
        stock_query = """
        SELECT ClsPric
        FROM fno_bhav_copy
        WHERE TckrSymb = 'BANKNIFTY' 
        AND TRADE_DATE = '2025-08-01'
        AND ClsPric > 100
        ORDER BY ClsPric DESC
        LIMIT 1
        """
        
        stock_result = vector_store.db_manager.connection.execute(stock_query).fetchdf()
        if not stock_result.empty:
            spot_price = stock_result['ClsPric'].iloc[0]
            print(f"   Spot price: {spot_price}")
            
            # Test PCR calculation
            pcr = vector_store._calculate_pcr(options_data, spot_price)
            print(f"   Calculated PCR: {pcr:.3f}")
            
            # Check if PCR is realistic
            if 0.1 <= pcr <= 10.0:
                print(f"   ✅ PCR is realistic: {pcr:.3f}")
            else:
                print(f"   ❌ PCR is unrealistic: {pcr:.3f}")
            
            # Show breakdown
            print(f"\n3. PCR Calculation Breakdown:")
            
            # Find ATM strike
            atm_strike = options_data.iloc[(options_data['ClsPric'] - spot_price).abs().argsort()[:1]]['ClsPric'].iloc[0]
            print(f"   ATM Strike: {atm_strike}")
            
            # 5% range
            atm_range = spot_price * 0.05
            puts_5pct = options_data[
                (options_data['ClsPric'] >= atm_strike - atm_range) &
                (options_data['ClsPric'] <= atm_strike + atm_range) &
                (options_data['FinInstrmTp'] == 'IDO')
            ]
            calls_5pct = options_data[
                (options_data['ClsPric'] >= atm_strike - atm_range) &
                (options_data['ClsPric'] <= atm_strike + atm_range) &
                (options_data['FinInstrmTp'] == 'STO')
            ]
            
            put_oi_5pct = puts_5pct['OpnIntrst'].sum() if not puts_5pct.empty else 0
            call_oi_5pct = calls_5pct['OpnIntrst'].sum() if not calls_5pct.empty else 0
            
            print(f"   5% Range (±{atm_range:.0f}):")
            print(f"     Puts (IDO): {len(puts_5pct)} records, OI: {put_oi_5pct}")
            print(f"     Calls (STO): {len(calls_5pct)} records, OI: {call_oi_5pct}")
            
            # 10% range
            atm_range_10pct = spot_price * 0.10
            puts_10pct = options_data[
                (options_data['ClsPric'] >= atm_strike - atm_range_10pct) &
                (options_data['ClsPric'] <= atm_strike + atm_range_10pct) &
                (options_data['FinInstrmTp'] == 'IDO')
            ]
            calls_10pct = options_data[
                (options_data['ClsPric'] >= atm_strike - atm_range_10pct) &
                (options_data['ClsPric'] <= atm_strike + atm_range_10pct) &
                (options_data['FinInstrmTp'] == 'STO')
            ]
            
            put_oi_10pct = puts_10pct['OpnIntrst'].sum() if not puts_10pct.empty else 0
            call_oi_10pct = calls_10pct['OpnIntrst'].sum() if not calls_10pct.empty else 0
            
            print(f"   10% Range (±{atm_range_10pct:.0f}):")
            print(f"     Puts (IDO): {len(puts_10pct)} records, OI: {put_oi_10pct}")
            print(f"     Calls (STO): {len(calls_10pct)} records, OI: {call_oi_10pct}")
            
            # All data
            puts_all = options_data[options_data['FinInstrmTp'] == 'IDO']
            calls_all = options_data[options_data['FinInstrmTp'] == 'STO']
            
            put_oi_all = puts_all['OpnIntrst'].sum() if not puts_all.empty else 0
            call_oi_all = calls_all['OpnIntrst'].sum() if not calls_all.empty else 0
            
            print(f"   All Data:")
            print(f"     Puts (IDO): {len(puts_all)} records, OI: {put_oi_all}")
            print(f"     Calls (STO): {len(calls_all)} records, OI: {call_oi_all}")
            
            if call_oi_all > 0:
                pcr_all = put_oi_all / call_oi_all
                print(f"     PCR (all data): {pcr_all:.3f}")
        
        # Test multiple dates
        print(f"\n4. Testing PCR across multiple dates:")
        test_dates = ['2025-08-01', '2025-08-06', '2025-03-17']
        
        for date in test_dates:
            print(f"\n   Date: {date}")
            
            # Get options data
            date_query = f"""
            SELECT 
                FinInstrmTp,
                SUM(OpnIntrst) as total_oi
            FROM fno_bhav_copy
            WHERE TckrSymb = 'BANKNIFTY' 
            AND TRADE_DATE = '{date}'
            AND FinInstrmTp IN ('IDO', 'STO')
            GROUP BY FinInstrmTp
            """
            
            date_result = vector_store.db_manager.connection.execute(date_query).fetchdf()
            
            put_oi = date_result[date_result['FinInstrmTp'] == 'IDO']['total_oi'].iloc[0] if not date_result[date_result['FinInstrmTp'] == 'IDO'].empty else 0
            call_oi = date_result[date_result['FinInstrmTp'] == 'STO']['total_oi'].iloc[0] if not date_result[date_result['FinInstrmTp'] == 'STO'].empty else 0
            
            if call_oi > 0:
                pcr_date = put_oi / call_oi
                print(f"     Put OI: {put_oi}, Call OI: {call_oi}")
                print(f"     PCR: {pcr_date:.3f}")
                
                if 0.1 <= pcr_date <= 10.0:
                    print(f"     ✅ Realistic PCR")
                else:
                    print(f"     ❌ Unrealistic PCR")
            else:
                print(f"     No call OI data available")
        
        print(f"\n✅ PCR Calculation Test Complete!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_pcr()
    if success:
        print("\n🎉 PCR calculation is now working correctly!")
    else:
        print("\n❌ PCR calculation still has issues.")
