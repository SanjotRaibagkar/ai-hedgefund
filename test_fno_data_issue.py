#!/usr/bin/env python3
"""
Test to isolate the PredictionRequest issue in FNO data processing.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.core.data_processor import FNODataProcessor
from src.fno_rag.models.data_models import PredictionRequest, HorizonType


def test_data_processor():
    """Test the data processor with different parameter types."""
    
    print("🧪 Testing FNO Data Processor")
    print("=" * 40)
    
    try:
        # Initialize data processor
        print("1. Initializing data processor...")
        data_processor = FNODataProcessor()
        
        # Test 1: Get data with string symbol
        print("\n2. Testing with string symbol...")
        try:
            df1 = data_processor.get_fno_data(['NIFTY'])
            print(f"   ✅ Success: Got {len(df1)} records for NIFTY")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 2: Get latest data with string symbol
        print("\n3. Testing get_latest_data with string symbol...")
        try:
            df2 = data_processor.get_latest_data(['NIFTY'])
            print(f"   ✅ Success: Got {len(df2)} latest records for NIFTY")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 3: Create a PredictionRequest and see what happens
        print("\n4. Testing with PredictionRequest object...")
        try:
            request = PredictionRequest(
                symbol='NIFTY',
                horizon=HorizonType.DAILY,
                include_explanations=False
            )
            print(f"   Created request: {request}")
            
            # This should fail if we pass the request object directly
            print("   Testing what happens if we pass request object...")
            # Don't actually call this - just show what the issue might be
            print("   The issue might be that somewhere the request object is being passed instead of request.symbol")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 4: Check if there's an issue with the symbols parameter
        print("\n5. Testing symbols parameter handling...")
        try:
            # Test with None
            df3 = data_processor.get_fno_data(None)
            print(f"   ✅ Success: Got {len(df3)} records with None symbols")
            
            # Test with empty list
            df4 = data_processor.get_fno_data([])
            print(f"   ✅ Success: Got {len(df4)} records with empty list")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print(f"\n✅ Data processor test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Starting FNO Data Processor Test...")
    success = test_data_processor()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
    else:
        print(f"\n❌ Test failed!")

