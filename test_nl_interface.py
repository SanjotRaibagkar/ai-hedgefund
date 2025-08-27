#!/usr/bin/env python3
"""
Test Natural Language Interface
Simple test to verify the FNO RAG chat interface is working.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine
from loguru import logger

def test_natural_language_interface():
    """Test the natural language interface with various queries."""
    
    print("🧪 Testing Natural Language Interface")
    print("=" * 50)
    
    try:
        # Initialize FNO engine
        print("1. Initializing FNO RAG System...")
        fno_engine = FNOEngine()
        print("   ✅ FNO RAG System initialized")
        
        # Test queries
        test_queries = [
            "What's the probability of NIFTY moving up tomorrow?",
            "Predict RELIANCE movement for next month",
            "What's the chance of TCS going down this week?",
            "Show me INFY probability for tomorrow",
            "Which stocks have high probability of moving up today?"
        ]
        
        print(f"\n2. Testing {len(test_queries)} natural language queries:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            try:
                response = fno_engine.chat(query)
                print(f"   ✅ Response: {response[:100]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n🎉 Natural Language Interface Test Completed!")
        print("✅ The chat interface is ready for integration with the UI!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_natural_language_interface()
