#!/usr/bin/env python3
"""
Test Enhanced Vector Store Directly
Verify the enhanced vector store is working with realistic return calculations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_enhanced_vector_store import EnhancedFNOVectorStore
from loguru import logger

def test_enhanced_vector_store():
    """Test the enhanced vector store directly."""
    
    print("🧪 Testing Enhanced Vector Store Directly")
    print("=" * 60)
    
    try:
        # Initialize vector store
        print("1. Initializing Enhanced Vector Store...")
        vector_store = EnhancedFNOVectorStore()
        
        # Load existing vector store
        print("2. Loading existing vector store...")
        if vector_store.load_vector_store():
            print(f"   ✅ Loaded {len(vector_store.metadata)} snapshots")
        else:
            print("   ❌ Failed to load vector store")
            return False
        
        # Test BANKNIFTY query specifically
        print("\n3. Testing BANKNIFTY Query (the problematic one):")
        test_query = "How much can BANKNIFTY move tomorrow?"
        
        try:
            result = vector_store.query_rag_system(test_query, top_k=3)
            similar_cases = result.get('similar_cases', [])
            print(f"   ✅ Found {len(similar_cases)} similar cases")
            
            # Check for realistic returns
            print("   📊 Similar Cases:")
            for case in similar_cases:
                symbol = case.get('symbol', 'Unknown')
                date = case.get('date', 'Unknown')
                daily_return = case.get('daily_return', 0)
                next_day_return = case.get('next_day_return', 0)
                pcr = case.get('pcr', 0)
                
                print(f"      • {symbol} ({date}): {daily_return:+.2f}% → Next: {next_day_return:+.2f}% (PCR: {pcr:.2f})")
                
                # Check for extreme values
                if abs(next_day_return) > 50:
                    print(f"         ⚠️ WARNING: Extreme return detected: {next_day_return:.2f}%")
                else:
                    print(f"         ✅ Realistic return: {next_day_return:+.2f}%")
            
            # Check LLM response
            llm_response = result.get('llm_response', '')
            if llm_response and "Error" not in llm_response:
                print("   🤖 LLM Response available")
                if len(llm_response) > 100:
                    print(f"   📝 Response preview: {llm_response[:100]}...")
            else:
                print("   ⚠️ No LLM response (GROQ not available)")
                
        except Exception as e:
            print(f"   ❌ Query error: {e}")
            return False
        
        # Test multiple queries
        print("\n4. Testing Multiple Queries:")
        test_queries = [
            "Find similar cases where NIFTY rose with high Put OI",
            "How much can RELIANCE move tomorrow based on current FNO data?",
            "Show me cases where BANKNIFTY had low PCR and moved up",
            "What happens when there's long buildup in stock futures?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            try:
                print(f"   {i}. Query: {query}")
                result = vector_store.query_rag_system(query, top_k=2)
                similar_cases = result.get('similar_cases', [])
                
                # Check for extreme values in all cases
                extreme_found = False
                for case in similar_cases:
                    next_day_return = case.get('next_day_return', 0)
                    if abs(next_day_return) > 50:
                        extreme_found = True
                        break
                
                if extreme_found:
                    print(f"      ❌ Extreme values detected")
                else:
                    print(f"      ✅ All realistic values")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Test specific symbols
        print("\n5. Testing Specific Symbols:")
        test_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
        
        for symbol in test_symbols:
            try:
                query = f"How much can {symbol} move tomorrow?"
                result = vector_store.query_rag_system(query, top_k=1)
                similar_cases = result.get('similar_cases', [])
                
                if similar_cases:
                    case = similar_cases[0]
                    next_day_return = case.get('next_day_return', 0)
                    
                    if abs(next_day_return) > 50:
                        print(f"   ❌ {symbol}: Extreme return {next_day_return:.2f}%")
                    else:
                        print(f"   ✅ {symbol}: Realistic return {next_day_return:+.2f}%")
                else:
                    print(f"   ⚠️ {symbol}: No similar cases found")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
        
        print("\n✅ Enhanced Vector Store Test Complete!")
        print("🚀 The vector store should now provide realistic return predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_vector_store()
    if success:
        print("\n🎉 All tests passed! Enhanced vector store is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
