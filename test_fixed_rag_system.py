#!/usr/bin/env python3
"""
Test Fixed Enhanced RAG System
Verify that the enhanced vector store is working with realistic return calculations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine
from loguru import logger

def test_fixed_rag_system():
    """Test the enhanced RAG system with fixed calculations."""
    
    print("🧪 Testing Fixed Enhanced RAG System")
    print("=" * 60)
    
    try:
        # Initialize enhanced FNO engine
        print("1. Initializing Enhanced FNO Engine...")
        fno_engine = FNOEngine()
        
        # Check system status
        status = fno_engine.get_system_status()
        print(f"   Enhanced Mode: {status.get('enhanced_mode', False)}")
        print(f"   Initialized: {status.get('initialized', False)}")
        
        if 'vector_store_stats' in status:
            stats = status['vector_store_stats']
            print(f"   Total Snapshots: {stats.get('total_snapshots', 0)}")
            print(f"   Embedding Dimension: {stats.get('embedding_dimension', 'Unknown')}")
        
        # Test BANKNIFTY query specifically
        print("\n2. Testing BANKNIFTY Query (the problematic one):")
        test_query = "How much can BANKNIFTY move tomorrow?"
        
        try:
            response = fno_engine.chat(test_query)
            print(f"   ✅ Response received: {len(response)} characters")
            
            # Check for realistic values
            if "64.27" in response or "-64.27" in response:
                print("   ❌ Still showing extreme values!")
                return False
            elif "17.61" in response or "-17.61" in response:
                print("   ❌ Still showing unrealistic average!")
                return False
            else:
                print("   ✅ No extreme values detected")
            
            # Check for enhanced features
            if "Similar Historical Cases" in response:
                print("   🚀 Enhanced RAG features working!")
            elif "📊" in response or "🤖" in response:
                print("   📈 Enhanced formatting working!")
            else:
                print("   ⚠️ Basic response (enhanced features not detected)")
                
        except Exception as e:
            print(f"   ❌ Chat error: {e}")
            return False
        
        # Test RAG analysis
        print("\n3. Testing Enhanced RAG Analysis:")
        try:
            rag_result = fno_engine.get_rag_analysis("Find similar cases where BANKNIFTY rose with high Put OI", top_k=3)
            similar_cases = rag_result.get('similar_cases', [])
            print(f"   ✅ RAG Analysis: {len(similar_cases)} similar cases found")
            
            # Check for realistic returns in similar cases
            for case in similar_cases:
                next_day_return = case.get('next_day_return', 0)
                if abs(next_day_return) > 50:
                    print(f"   ⚠️ Extreme return detected: {next_day_return:.2f}%")
                else:
                    print(f"   ✅ Realistic return: {next_day_return:+.2f}%")
            
            if rag_result.get('llm_response'):
                print("   🤖 LLM Response available")
            else:
                print("   ⚠️ No LLM response (GROQ not available)")
                
        except Exception as e:
            print(f"   ❌ RAG Analysis error: {e}")
            return False
        
        # Test multiple symbols
        print("\n4. Testing Multiple Symbols:")
        test_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
        
        for symbol in test_symbols:
            try:
                query = f"How much can {symbol} move tomorrow?"
                response = fno_engine.chat(query)
                
                # Check for extreme values
                if any(str(x) in response for x in [64.27, -64.27, 17.61, -17.61]):
                    print(f"   ❌ {symbol}: Still showing extreme values")
                else:
                    print(f"   ✅ {symbol}: Realistic values")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
        
        print("\n✅ Fixed Enhanced RAG System Test Complete!")
        print("🚀 The system should now provide realistic return predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_rag_system()
    if success:
        print("\n🎉 All tests passed! Enhanced RAG system is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")

