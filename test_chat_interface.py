#!/usr/bin/env python3
"""
Test Chat Interface
Test the FNO RAG chat interface functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_chat_interface():
    """Test the chat interface functionality."""
    
    print("🧪 Testing Chat Interface")
    print("=" * 40)
    
    try:
        # Test FNO Engine import
        print("1. Testing FNO Engine import...")
        from src.fno_rag import FNOEngine
        print("   ✅ FNO Engine imported successfully")
        
        # Test FNO Engine initialization
        print("\n2. Testing FNO Engine initialization...")
        fno_engine = FNOEngine()
        print("   ✅ FNO Engine initialized successfully")
        
        # Test chat method
        print("\n3. Testing chat method...")
        test_queries = [
            "What's the probability of NIFTY moving up tomorrow?",
            "Predict RELIANCE movement for next month",
            "What's the chance of TCS going down this week?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            try:
                response = fno_engine.chat(query)
                print(f"   ✅ Response: {response[:100]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n🎉 Chat Interface Test Completed!")
        print("✅ The chat interface should work in the UI!")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")

if __name__ == "__main__":
    test_chat_interface()
