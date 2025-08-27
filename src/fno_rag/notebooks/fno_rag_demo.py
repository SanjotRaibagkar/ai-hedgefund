#!/usr/bin/env python3
"""
FNO RAG System Demo Script
Demonstrates the key features of the FNO RAG system.
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.fno_rag import FNOEngine, HorizonType
from src.fno_rag.models.data_models import PredictionRequest


def main():
    """Main demo function."""
    print("🚀 FNO RAG System Demo")
    print("=" * 50)
    
    # Initialize the FNO engine
    print("\n1. Initializing FNO RAG System...")
    groq_api_key = os.getenv('GROQ_API_KEY')
    fno_engine = FNOEngine(groq_api_key=groq_api_key)
    
    # Check system status
    status = fno_engine.get_system_status()
    print(f"✅ System initialized: {status.get('initialized', False)}")
    
    # Get available symbols
    print("\n2. Getting available symbols...")
    symbols = fno_engine.get_available_symbols()
    print(f"📊 Available symbols: {len(symbols)}")
    print(f"Sample: {symbols[:5]}")
    
    # Example 1: Single stock prediction
    print("\n3. Single Stock Prediction Example")
    print("-" * 40)
    
    try:
        result = fno_engine.predict_probability("RELIANCE", HorizonType.DAILY)
        print(f"📈 {result.symbol} (Daily):")
        print(f"   Up: {result.up_probability:.1%}")
        print(f"   Down: {result.down_probability:.1%}")
        print(f"   Neutral: {result.neutral_probability:.1%}")
        print(f"   Confidence: {result.confidence_score:.1%}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Batch prediction
    print("\n4. Batch Prediction Example")
    print("-" * 40)
    
    try:
        test_symbols = ["RELIANCE", "TCS", "INFY"]
        results = fno_engine.predict_batch(test_symbols, HorizonType.WEEKLY)
        
        for result in results:
            print(f"📊 {result.symbol} (Weekly):")
            print(f"   Up: {result.up_probability:.1%}")
            print(f"   Down: {result.down_probability:.1%}")
            print(f"   Neutral: {result.neutral_probability:.1%}")
            print(f"   Confidence: {result.confidence_score:.1%}")
            print()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Stock search
    print("\n5. Stock Search Example")
    print("-" * 40)
    
    try:
        results = fno_engine.search_stocks(
            query="up tomorrow",
            horizon=HorizonType.DAILY,
            min_probability=0.3,
            max_results=3
        )
        
        print(f"🔍 Found {len(results)} stocks with high up probability:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.symbol}: Up={result.up_probability:.1%}, Confidence={result.confidence_score:.1%}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Natural language queries
    print("\n6. Natural Language Query Examples")
    print("-" * 40)
    
    queries = [
        "What's the probability of RELIANCE moving up tomorrow?",
        "Find FNO stocks that can move 3% tomorrow",
        "What's the system status?"
    ]
    
    for query in queries:
        print(f"\n🤖 Query: {query}")
        try:
            response = fno_engine.chat_query(query)
            if response.get('error'):
                print(f"   ❌ Error: {response['message']}")
            else:
                print(f"   ✅ Response: {response['message'][:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Example 5: Export results
    print("\n7. Export Results Example")
    print("-" * 40)
    
    try:
        results = fno_engine.predict_batch(["RELIANCE", "TCS"], HorizonType.DAILY)
        
        # Export to JSON
        json_data = fno_engine.export_results(results, 'json')
        print(f"📄 JSON export length: {len(json_data)} characters")
        
        # Export to CSV
        csv_data = fno_engine.export_results(results, 'csv')
        print(f"📊 CSV export length: {len(csv_data)} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Demo completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Probability prediction for different horizons")
    print("✅ Batch processing of multiple stocks")
    print("✅ Stock search based on criteria")
    print("✅ Natural language query interface")
    print("✅ Result export in multiple formats")
    print("✅ System status monitoring")


if __name__ == "__main__":
    main()
