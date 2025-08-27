#!/usr/bin/env python3
"""
Debug script to check data types in FNO data.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.core.data_processor import FNODataProcessor


def debug_data_types():
    """Debug the data types in FNO data."""
    
    print("🔍 Debugging FNO Data Types")
    print("=" * 40)
    
    try:
        # Initialize data processor
        print("1. Initializing data processor...")
        data_processor = FNODataProcessor()
        
        # Get a small sample of data
        print("\n2. Getting sample data...")
        df = data_processor.get_fno_data(['NIFTY'])
        # Take only first 10 rows
        df = df.head(10)
        
        print(f"\n3. DataFrame info:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Data types:")
        for col, dtype in df.dtypes.items():
            print(f"     {col}: {dtype}")
        
        print(f"\n4. Sample data:")
        print(df.head())
        
        print(f"\n5. Testing groupby operation...")
        try:
            # Test the problematic operation
            result = df.groupby('symbol')['close_price'].pct_change()
            print(f"   ✅ Groupby operation successful: {len(result)} results")
        except Exception as e:
            print(f"   ❌ Groupby operation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Debug completed!")
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 Starting FNO Data Type Debug...")
    success = debug_data_types()
    
    if success:
        print(f"\n🎉 Debug completed successfully!")
    else:
        print(f"\n❌ Debug failed!")
