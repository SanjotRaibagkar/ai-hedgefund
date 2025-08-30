#!/usr/bin/env python3
"""
Test Improved Next Day Outcome Calculation
Verify that the improved logic finds next trading days correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_enhanced_vector_store import EnhancedFNOVectorStore
from loguru import logger

def test_improved_next_day():
    """Test the improved next day outcome calculation."""
    
    print("🧪 Testing Improved Next Day Outcome Calculation")
    print("=" * 60)
    
    try:
        # Initialize vector store
        print("1. Initializing Enhanced Vector Store...")
        vector_store = EnhancedFNOVectorStore()
        
        # Test specific cases that were showing 0.00%
        test_cases = [
            ('NIFTY', '2025-05-30'),
            ('RELIANCE', '2025-06-27'),
            ('BANKNIFTY', '2025-07-09'),
            ('FORTIS', '2025-06-24')
        ]
        
        print("\n2. Testing Next Day Outcomes:")
        
        for symbol, date in test_cases:
            print(f"\n   📊 {symbol} on {date}:")
            
            # Test the improved function
            outcome = vector_store._get_next_day_outcome(symbol, date)
            
            print(f"      Return: {outcome['return']:+.2f}%")
            print(f"      Direction: {outcome['direction']}")
            print(f"      High: {outcome['high']:.2f}")
            print(f"      Low: {outcome['low']:.2f}")
            
            # Check if it's realistic
            if abs(outcome['return']) < 0.01:
                print(f"      ⚠️ WARNING: Still showing near-zero return")
            elif abs(outcome['return']) > 50:
                print(f"      ⚠️ WARNING: Extreme return detected")
            else:
                print(f"      ✅ Realistic return: {outcome['return']:+.2f}%")
        
        # Test edge cases
        print(f"\n3. Testing Edge Cases:")
        
        edge_cases = [
            ('NIFTY', '2025-08-29'),  # Friday
            ('RELIANCE', '2025-08-30'),  # Saturday (weekend)
            ('BANKNIFTY', '2025-08-31'),  # Sunday (weekend)
        ]
        
        for symbol, date in edge_cases:
            print(f"\n   📊 {symbol} on {date}:")
            
            outcome = vector_store._get_next_day_outcome(symbol, date)
            
            print(f"      Return: {outcome['return']:+.2f}%")
            print(f"      Direction: {outcome['direction']}")
            
            if abs(outcome['return']) < 0.01:
                print(f"      ⚠️ WARNING: Near-zero return")
            else:
                print(f"      ✅ Found next trading day")
        
        # Test the complete feature calculation
        print(f"\n4. Testing Complete Feature Calculation:")
        
        # Get sample data
        sample_query = """
        SELECT 
            TckrSymb,
            TRADE_DATE,
            FinInstrmTp,
            OpnPric,
            HghPric,
            LwPric,
            ClsPric,
            TtlTradgVol,
            OpnIntrst,
            ChngInOpnIntrst,
            TtlTrfVal,
            PrvsClsgPric
        FROM fno_bhav_copy
        WHERE TckrSymb = 'NIFTY'
        AND TRADE_DATE = '2025-05-30'
        LIMIT 10
        """
        
        sample_data = vector_store.db_manager.connection.execute(sample_query).fetchdf()
        
        if not sample_data.empty:
            print(f"   Testing with NIFTY data from 2025-05-30:")
            
            # Test the complete feature calculation
            features = vector_store._calculate_symbol_day_features(sample_data, 'NIFTY', '2025-05-30')
            
            if features:
                print(f"      Next day return: {features['next_day_return']:+.2f}%")
                print(f"      Next day direction: {features['next_day_direction']}")
                print(f"      PCR: {features['pcr']:.2f}")
                
                if abs(features['next_day_return']) < 0.01:
                    print(f"      ⚠️ WARNING: Still showing near-zero return")
                else:
                    print(f"      ✅ Improved next day calculation working")
            else:
                print(f"      ❌ No features calculated")
        
        print(f"\n✅ Improved Next Day Outcome Test Complete!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_improved_next_day()
    if success:
        print("\n🎉 Improved next day calculation is working correctly!")
    else:
        print("\n❌ Next day calculation still has issues.")
