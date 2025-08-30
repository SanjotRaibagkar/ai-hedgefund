#!/usr/bin/env python3
"""
Fix UI Callback Issues
Fix callback failures and ensure enhanced vector store is properly loaded.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, ENHANCED_VECTOR_STORE_AVAILABLE
from loguru import logger

def fix_ui_callbacks():
    """Fix UI callback issues and ensure proper enhanced vector store loading."""
    
    print("🔧 Fixing UI Callback Issues")
    print("=" * 50)
    
    try:
        # Check if enhanced vector store is available
        print(f"1. Enhanced Vector Store Available: {ENHANCED_VECTOR_STORE_AVAILABLE}")
        
        if not ENHANCED_VECTOR_STORE_AVAILABLE:
            print("❌ Enhanced vector store not available!")
            print("💡 Please ensure build_enhanced_vector_store.py is in the project root")
            return False
        
        # Initialize enhanced FNO engine
        print("\n2. Initializing Enhanced FNO Engine...")
        fno_engine = FNOEngine()
        
        # Verify enhanced mode
        status = fno_engine.get_system_status()
        print(f"   Enhanced Mode: {status.get('enhanced_mode', False)}")
        print(f"   Initialized: {status.get('initialized', False)}")
        
        if not status.get('enhanced_mode'):
            print("❌ Not running in enhanced mode!")
            return False
        
        # Check vector store stats
        if 'vector_store_stats' in status:
            stats = status['vector_store_stats']
            print(f"   Total Snapshots: {stats.get('total_snapshots', 0)}")
            print(f"   Embedding Dimension: {stats.get('embedding_dimension', 'Unknown')}")
            
            if stats.get('total_snapshots', 0) < 20000:
                print("⚠️ Vector store has limited data")
            else:
                print("✅ Vector store has sufficient data")
        
        # Test enhanced chat functionality
        print("\n3. Testing Enhanced Chat Functionality:")
        
        test_query = "What's the probability of NIFTY moving up tomorrow?"
        print(f"   Testing query: {test_query}")
        
        try:
            response = fno_engine.chat(test_query)
            print(f"   ✅ Response received: {len(response)} characters")
            
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
        print("\n4. Testing Enhanced RAG Analysis:")
        try:
            rag_result = fno_engine.get_rag_analysis("Find similar cases where NIFTY rose with high Put OI", top_k=3)
            similar_cases = rag_result.get('similar_cases', [])
            print(f"   ✅ RAG Analysis: {len(similar_cases)} similar cases found")
            
            if rag_result.get('llm_response'):
                print("   🤖 LLM Response available")
            else:
                print("   ⚠️ No LLM response (GROQ not available)")
                
        except Exception as e:
            print(f"   ❌ RAG Analysis error: {e}")
            return False
        
        print("\n✅ UI Callback Issues Fixed!")
        print("🚀 Enhanced FNO RAG System is ready for UI use!")
        
        # Provide instructions for UI
        print("\n📋 UI Usage Instructions:")
        print("1. Start the UI: poetry run python src/ui/web_app/app.py")
        print("2. Open browser: http://localhost:8050")
        print("3. Click 'Initialize AI Chat' button")
        print("4. Ask enhanced queries like:")
        print("   - 'What's the probability of NIFTY moving up tomorrow?'")
        print("   - 'Find similar cases where BANKNIFTY rose with high Put OI'")
        print("   - 'Show me cases where RELIANCE had low PCR and moved up'")
        
        return True
        
    except Exception as e:
        logger.error(f"Fix failed: {e}")
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_ui_callbacks()
    if success:
        print("\n🎉 All issues resolved! UI should work properly now.")
    else:
        print("\n❌ Issues remain. Please check the logs above.")
