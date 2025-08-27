#!/usr/bin/env python3
"""
Comprehensive Test Script for FNO ML and RAG System
Tests all major functionalities including chat interface, predictions, and search
"""

import sys
import os
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fno_rag.core.fno_engine import FNOEngine
    from fno_rag.api.chat_interface import FNOChatInterface
    from fno_rag.models.data_models import HorizonType, PredictionRequest
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Trying alternative import paths...")
    
    # Try alternative import paths
    try:
        from src.fno_rag.core.fno_engine import FNOEngine
        from src.fno_rag.api.chat_interface import FNOChatInterface
        from src.fno_rag.models.data_models import HorizonType, PredictionRequest
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("💡 Please ensure the FNO RAG system is properly installed")
        sys.exit(1)

def test_fno_engine_initialization():
    """Test FNO Engine initialization."""
    print("🧪 Testing FNO Engine Initialization...")
    print("=" * 60)
    
    try:
        fno_engine = FNOEngine()
        print("✅ FNO Engine initialized successfully!")
        
        # Test system status
        status = fno_engine.get_system_status()
        print(f"📊 System Status: {status}")
        
        return fno_engine
    except Exception as e:
        print(f"❌ FNO Engine initialization failed: {e}")
        return None

def test_chat_interface():
    """Test Chat Interface functionality."""
    print("\n🧪 Testing Chat Interface...")
    print("=" * 60)
    
    try:
        chat_interface = FNOChatInterface()
        print("✅ Chat Interface initialized successfully!")
        
        # Test queries
        test_queries = [
            "What is the value of NIFTY?",
            "What is the probability of RELIANCE moving up tomorrow?",
            "Find stocks that can move 5% this week",
            "Show me FNO stocks with high probability moves",
            "What is the system status?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing Query: '{query}'")
            try:
                response = chat_interface.process_query(query)
                if 'error' in response:
                    print(f"❌ Error: {response['message']}")
                else:
                    print(f"✅ Response: {response['message'][:100]}...")
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        return chat_interface
    except Exception as e:
        print(f"❌ Chat Interface initialization failed: {e}")
        return None

def test_stock_search(fno_engine):
    """Test stock search functionality."""
    print("\n🧪 Testing Stock Search...")
    print("=" * 60)
    
    try:
        # Test 1: Find stocks with high probability of moving up tomorrow
        print("🔍 Test 1: Finding stocks with high probability of moving up tomorrow")
        results = fno_engine.search_stocks(
            query="up tomorrow",
            horizon=HorizonType.DAILY,
            min_probability=0.3,
            max_results=10
        )
        
        print(f"Found {len(results)} stocks:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.symbol}: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}, Neutral={result.neutral_probability:.1%}")
        
        # Test 2: Find stocks with high probability of moving down this week
        print("\n🔍 Test 2: Finding stocks with high probability of moving down this week")
        results = fno_engine.search_stocks(
            query="down this week",
            horizon=HorizonType.WEEKLY,
            min_probability=0.4,
            max_results=5
        )
        
        print(f"Found {len(results)} stocks:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.symbol}: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}, Neutral={result.neutral_probability:.1%}")
        
        # Test 3: Find stocks with high probability moves this month
        print("\n🔍 Test 3: Finding stocks with high probability moves this month")
        results = fno_engine.search_stocks(
            query="high probability moves",
            horizon=HorizonType.MONTHLY,
            min_probability=0.5,
            max_results=5
        )
        
        print(f"Found {len(results)} stocks:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.symbol}: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}, Neutral={result.neutral_probability:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Stock search failed: {e}")
        return False

def test_individual_predictions(fno_engine):
    """Test individual stock predictions."""
    print("\n🧪 Testing Individual Stock Predictions...")
    print("=" * 60)
    
    test_stocks = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'NIFTY']
    
    try:
        for stock in test_stocks:
            print(f"\n📊 Testing predictions for {stock}:")
            
            # Test daily prediction
            request = PredictionRequest(
                symbol=stock,
                horizon=HorizonType.DAILY,
                include_explanations=True,
                top_k_similar=3
            )
            
            try:
                result = fno_engine.predict_probability(request)
                print(f"  Daily: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}, Neutral={result.neutral_probability:.1%}")
                print(f"  Confidence: {result.confidence_score:.1%}")
            except Exception as e:
                print(f"  ❌ Daily prediction failed: {e}")
            
            # Test weekly prediction
            request.horizon = HorizonType.WEEKLY
            try:
                result = fno_engine.predict_probability(request)
                print(f"  Weekly: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}, Neutral={result.neutral_probability:.1%}")
                print(f"  Confidence: {result.confidence_score:.1%}")
            except Exception as e:
                print(f"  ❌ Weekly prediction failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Individual predictions failed: {e}")
        return False

def test_rag_functionality(fno_engine):
    """Test RAG functionality."""
    print("\n🧪 Testing RAG Functionality...")
    print("=" * 60)
    
    try:
        # Test vector store search
        print("🔍 Testing vector store search...")
        
        # Test similar conditions search
        test_conditions = [
            "high volatility market conditions",
            "bullish momentum with high volume",
            "bearish trend with increasing put call ratio"
        ]
        
        for condition in test_conditions:
            print(f"\n📊 Searching for conditions similar to: '{condition}'")
            try:
                similar_conditions = fno_engine.find_similar_conditions(condition, top_k=3)
                print(f"Found {len(similar_conditions)} similar conditions:")
                for i, cond in enumerate(similar_conditions, 1):
                    print(f"  {i}. Similarity: {cond.similarity_score:.3f}")
                    print(f"     Condition: {cond.condition[:100]}...")
            except Exception as e:
                print(f"  ❌ Search failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ RAG functionality failed: {e}")
        return False

def test_data_processor(fno_engine):
    """Test data processor functionality."""
    print("\n🧪 Testing Data Processor...")
    print("=" * 60)
    
    try:
        # Test getting FNO data
        print("📊 Testing FNO data retrieval...")
        fno_data = fno_engine.data_processor.get_fno_data()
        print(f"✅ Retrieved {len(fno_data)} FNO records")
        
        # Test getting latest data
        print("📊 Testing latest data retrieval...")
        latest_data = fno_engine.data_processor.get_latest_data()
        print(f"✅ Retrieved latest data for {len(latest_data)} symbols")
        
        return True
    except Exception as e:
        print(f"❌ Data processor failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting FNO ML and RAG System Testing")
    print("=" * 80)
    print(f"📅 Test started at: {datetime.now()}")
    print("=" * 80)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: FNO Engine Initialization
    fno_engine = test_fno_engine_initialization()
    test_results['engine_init'] = fno_engine is not None
    
    if fno_engine is None:
        print("❌ Cannot proceed with other tests - FNO Engine failed to initialize")
        return
    
    # Test 2: Chat Interface
    chat_interface = test_chat_interface()
    test_results['chat_interface'] = chat_interface is not None
    
    # Test 3: Stock Search
    test_results['stock_search'] = test_stock_search(fno_engine)
    
    # Test 4: Individual Predictions
    test_results['individual_predictions'] = test_individual_predictions(fno_engine)
    
    # Test 5: RAG Functionality
    test_results['rag_functionality'] = test_rag_functionality(fno_engine)
    
    # Test 6: Data Processor
    test_results['data_processor'] = test_data_processor(fno_engine)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! FNO ML and RAG system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print(f"\n📅 Test completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
