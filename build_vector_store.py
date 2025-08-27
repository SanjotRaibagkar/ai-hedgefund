#!/usr/bin/env python3
"""
Build FNO Vector Store
Build the RAG vector store with historical FNO data for similarity search.
"""

import sys
import os
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, HorizonType
from loguru import logger


def build_vector_store():
    """Build the FNO vector store with historical data."""
    
    print("🏗️ Building FNO Vector Store")
    print("=" * 50)
    
    try:
        # Initialize FNO engine
        print("1. Initializing FNO RAG System...")
        start_time = time.time()
        fno_engine = FNOEngine()
        init_time = time.time() - start_time
        print(f"   ✅ Initialized in {init_time:.2f} seconds")
        
        # Build vector store
        print("\n2. Building vector store...")
        vector_start = time.time()
        
        # Use last 3 months of data for vector store
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        print(f"   📅 Using data from {start_date} to {end_date}")
        print("   ⏳ This may take several minutes...")
        
        try:
            stats = fno_engine.build_vector_store(start_date=start_date, end_date=end_date)
            vector_time = time.time() - vector_start
            
            print(f"   ✅ Vector store built successfully in {vector_time:.2f} seconds")
            print(f"   📊 Stats: {stats}")
            
        except Exception as e:
            print(f"   ❌ Failed to build vector store: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n3. Vector store building completed!")
        
        # Test vector store
        print("\n4. Testing vector store...")
        try:
            # Test with a sample symbol
            test_symbol = 'NIFTY'
            test_horizon = HorizonType.DAILY
            
            print(f"   🔍 Testing similarity search for {test_symbol}...")
            
            # This will test if the vector store is working
            result = fno_engine.predict_probability(test_symbol, test_horizon)
            
            if result:
                print(f"   ✅ Vector store test successful")
                print(f"   📈 {test_symbol} prediction: Up={result.probabilities['up']:.3f}, Down={result.probabilities['down']:.3f}")
            else:
                print(f"   ⚠️ Vector store test returned no result")
                
        except Exception as e:
            print(f"   ❌ Vector store test failed: {e}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Vector store building completed in {total_time:.2f} seconds!")
        return True
        
    except Exception as e:
        print(f"❌ Vector store building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🏗️ Starting FNO Vector Store Building...")
    success = build_vector_store()
    
    if success:
        print(f"\n🎉 Vector store building completed successfully!")
    else:
        print(f"\n❌ Vector store building failed!")
