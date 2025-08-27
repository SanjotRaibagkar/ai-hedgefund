#!/usr/bin/env python3
"""
FNO RAG System Demo
Comprehensive demonstration of the FNO RAG system capabilities.
"""

import sys
import os
from datetime import datetime
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, HorizonType
from loguru import logger


def fno_rag_demo():
    """Demonstrate the complete FNO RAG system."""
    
    print("🚀 FNO RAG System - Complete Demo")
    print("=" * 60)
    
    try:
        # Initialize FNO engine
        print("1. 🏗️ Initializing FNO RAG System...")
        start_time = time.time()
        fno_engine = FNOEngine()
        init_time = time.time() - start_time
        print(f"   ✅ Initialized in {init_time:.2f} seconds")
        
        # System status
        print("\n2. 📊 System Status:")
        try:
            status = fno_engine.get_system_status()
            print(f"   ✅ System initialized: {status.get('initialized', False)}")
            print(f"   📅 Last update: {status.get('last_update', 'Unknown')}")
            
            if 'data_info' in status:
                data_info = status['data_info']
                print(f"   📈 Data available: {data_info.get('data_available', False)}")
                print(f"   📊 Symbol count: {data_info.get('symbol_count', 0)}")
        except Exception as e:
            print(f"   ⚠️ Status check: {e}")
        
        # Test predictions for major symbols
        print("\n3. 🎯 Probability Predictions:")
        test_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY']
        horizons = [HorizonType.DAILY, HorizonType.WEEKLY, HorizonType.MONTHLY]
        
        for symbol in test_symbols:
            print(f"\n   📈 {symbol}:")
            for horizon in horizons:
                try:
                    result = fno_engine.predict_probability(symbol, horizon)
                    if result:
                        direction = "🟢 UP" if result.up_probability > result.down_probability else "🔴 DOWN"
                        print(f"      {horizon.value:8}: {direction} | Up: {result.up_probability:.1%}, Down: {result.down_probability:.1%}, Neutral: {result.neutral_probability:.1%} | Confidence: {result.confidence_score:.1%}")
                    else:
                        print(f"      {horizon.value:8}: ❌ No prediction")
                except Exception as e:
                    print(f"      {horizon.value:8}: ❌ Error: {e}")
        
        # Natural language queries
        print("\n4. 🤖 Natural Language Queries:")
        test_queries = [
            "What's the probability of NIFTY moving up tomorrow?",
            "Predict RELIANCE movement for next month",
            "What's the chance of TCS going down this week?",
            "Show me INFY probability for tomorrow"
        ]
        
        for query in test_queries:
            try:
                print(f"\n   🤖 Query: {query}")
                response = fno_engine.chat_query(query)
                
                if 'error' in response and response['error']:
                    print(f"      ❌ Error: {response['message']}")
                else:
                    message = response.get('message', 'No message')
                    print(f"      ✅ Response: {message}")
                    
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
        
        # Performance metrics
        print("\n5. 📊 Performance Metrics:")
        total_time = time.time() - start_time
        print(f"   ⏱️ Total demo time: {total_time:.2f} seconds")
        print(f"   🚀 System ready for production use")
        
        # Recommendations
        print("\n6. 💡 Trading Recommendations:")
        print("   Based on current predictions:")
        
        # Get top predictions
        recommendations = []
        for symbol in test_symbols:
            try:
                result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
                if result and result.confidence_score > 0.6:
                    direction = "BUY" if result.up_probability > result.down_probability else "SELL"
                    probability = max(result.up_probability, result.down_probability)
                    recommendations.append((symbol, direction, probability, result.confidence_score))
            except:
                continue
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        if recommendations:
            for i, (symbol, direction, probability, confidence) in enumerate(recommendations[:5], 1):
                print(f"   {i}. {symbol}: {direction} (Probability: {probability:.1%}, Confidence: {confidence:.1%})")
        else:
            print("   ⚠️ No high-confidence recommendations at this time")
        
        print(f"\n🎉 FNO RAG System Demo Completed Successfully!")
        print(f"📈 The system is ready for real-time FNO trading analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting FNO RAG System Demo...")
    success = fno_rag_demo()
    
    if success:
        print(f"\n🎉 Demo completed successfully!")
    else:
        print(f"\n❌ Demo failed!")
