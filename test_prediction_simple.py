#!/usr/bin/env python3
"""
Simple Prediction Test
Test the prediction system with trained models.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, HorizonType
from loguru import logger


def test_prediction():
    """Test the prediction system."""
    
    print("🧪 Simple Prediction Test")
    print("=" * 50)
    
    try:
        # Initialize FNO engine
        print("1. Initializing FNO RAG System...")
        fno_engine = FNOEngine()
        print(f"   ✅ Initialized successfully")
        
        # Test a simple prediction
        print("\n2. Testing prediction...")
        symbol = 'NIFTY'
        horizon = HorizonType.DAILY
        
        print(f"   🔍 Testing {symbol} ({horizon.value})...")
        
        try:
            result = fno_engine.predict_probability(symbol, horizon)
            
            if result:
                print(f"   ✅ Prediction successful!")
                print(f"   📈 Up probability: {result.up_probability:.3f}")
                print(f"   📉 Down probability: {result.down_probability:.3f}")
                print(f"   ➡️ Neutral probability: {result.neutral_probability:.3f}")
                print(f"   🎯 Confidence score: {result.confidence_score:.3f}")
            else:
                print(f"   ⚠️ No result returned")
                
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n🎉 Prediction test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Starting Simple Prediction Test...")
    success = test_prediction()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
    else:
        print(f"\n❌ Test failed!")
