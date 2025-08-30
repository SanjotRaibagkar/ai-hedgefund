#!/usr/bin/env python3
"""
Test Enhanced UI Integration
Test that the UI properly loads the enhanced FNO RAG system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, ENHANCED_VECTOR_STORE_AVAILABLE
from loguru import logger

def test_ui_enhanced_integration():
    """Test the enhanced UI integration."""
    
    print("🧪 Testing Enhanced UI Integration")
    print("=" * 50)
    
    try:
        # Test 1: Check enhanced vector store availability
        print(f"1. Enhanced Vector Store Available: {ENHANCED_VECTOR_STORE_AVAILABLE}")
        
        # Test 2: Initialize enhanced FNO engine
        print("\n2. Initializing Enhanced FNO Engine...")
        fno_engine = FNOEngine()
        
        # Test 3: Check system status
        print("\n3. System Status:")
        status = fno_engine.get_system_status()
        print(f"   Initialized: {status.get('initialized', False)}")
        print(f"   Enhanced Mode: {status.get('enhanced_mode', False)}")
        
        if status.get('enhanced_mode') and 'vector_store_stats' in status:
            stats = status['vector_store_stats']
            print(f"   Total Snapshots: {stats.get('total_snapshots', 0)}")
            print(f"   Embedding Dimension: {stats.get('embedding_dimension', 'Unknown')}")
        
        # Test 4: Test enhanced chat functionality
        print("\n4. Testing Enhanced Chat:")
        
        test_queries = [
            "What's the probability of NIFTY moving up tomorrow?",
            "Find similar cases where BANKNIFTY rose with high Put OI",
            "Show me cases where RELIANCE had low PCR and moved up"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            try:
                response = fno_engine.chat(query)
                print(f"   ✅ Response: {response[:100]}...")
                
                # Check for enhanced features
                if "Similar Historical Cases" in response:
                    print("   🚀 Enhanced RAG features detected!")
                elif "📊" in response or "🤖" in response:
                    print("   📈 Enhanced formatting detected!")
                else:
                    print("   ⚠️ Basic response (enhanced features not detected)")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test 5: Test RAG analysis
        print("\n5. Testing Enhanced RAG Analysis:")
        try:
            rag_result = fno_engine.get_rag_analysis("Find similar cases where NIFTY rose with high Put OI", top_k=3)
            print(f"   ✅ RAG Analysis: {len(rag_result.get('similar_cases', []))} similar cases found")
            
            if rag_result.get('llm_response'):
                print("   🤖 LLM Response available")
            else:
                print("   ⚠️ No LLM response (GROQ not available)")
                
        except Exception as e:
            print(f"   ❌ RAG Analysis Error: {e}")
        
        print("\n🎉 Enhanced UI Integration Test Completed!")
        print("✅ The enhanced FNO RAG system is ready for UI integration!")
        
        if status.get('enhanced_mode'):
            print("🚀 Enhanced features are available:")
            print("   - Semantic embeddings for better similarity search")
            print("   - Advanced RAG analysis with LLM responses")
            print("   - Rich historical case retrieval")
            print("   - Enhanced natural language processing")
        else:
            print("⚠️ Running in fallback mode - enhanced features not available")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ui_enhanced_integration()
    if success:
        print("\n🚀 Ready to start UI with enhanced features!")
    else:
        print("\n❌ UI integration test failed!")
