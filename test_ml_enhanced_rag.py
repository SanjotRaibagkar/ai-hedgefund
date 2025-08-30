#!/usr/bin/env python3
"""
Test ML + Enhanced RAG System
Comprehensive testing of ML predictions and RAG analysis with CSV output
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import time
from typing import Dict, List, Any
import csv

# Import the enhanced FNO system
from src.fno_rag import FNOEngine, HorizonType
from build_enhanced_vector_store import EnhancedFNOVectorStore

class MLEnhancedRAGTester:
    """Test ML + Enhanced RAG system comprehensively."""
    
    def __init__(self):
        """Initialize the tester."""
        self.fno_engine = FNOEngine()
        self.vector_store = EnhancedFNOVectorStore()
        self.results = []
        self.test_symbols = [
            'NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY', 
            'HDFC', 'ICICIBANK', 'ITC', 'SBIN', 'BHARTIARTL'
        ]
        
    def test_ml_predictions(self) -> List[Dict]:
        """Test ML predictions for different symbols and horizons."""
        logger.info("🧪 Testing ML Predictions...")
        
        ml_results = []
        horizons = [HorizonType.DAILY, HorizonType.WEEKLY, HorizonType.MONTHLY]
        
        for symbol in self.test_symbols:
            for horizon in horizons:
                try:
                    start_time = time.time()
                    
                    # Get ML prediction
                    prediction = self.fno_engine.predict_probability(symbol, horizon)
                    
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    result = {
                        'test_type': 'ML_Prediction',
                        'symbol': symbol,
                        'horizon': horizon.value,
                        'probability': prediction.get('probability', 0.0),
                        'confidence': prediction.get('confidence', 0.0),
                        'direction': prediction.get('direction', 'UNKNOWN'),
                        'response_time_ms': round(response_time, 2),
                        'status': 'SUCCESS',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    ml_results.append(result)
                    logger.info(f"✅ {symbol} {horizon.value}: {prediction.get('probability', 0.0):.2%}")
                    
                except Exception as e:
                    error_result = {
                        'test_type': 'ML_Prediction',
                        'symbol': symbol,
                        'horizon': horizon.value,
                        'probability': 0.0,
                        'confidence': 0.0,
                        'direction': 'ERROR',
                        'response_time_ms': 0.0,
                        'status': 'ERROR',
                        'error_message': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    ml_results.append(error_result)
                    logger.error(f"❌ {symbol} {horizon.value}: {e}")
        
        return ml_results
    
    def test_rag_analysis(self) -> List[Dict]:
        """Test RAG analysis with different queries."""
        logger.info("🧪 Testing RAG Analysis...")
        
        rag_results = []
        test_queries = [
            "What's the probability of NIFTY moving up tomorrow?",
            "Find similar cases where BANKNIFTY rose with high Put OI",
            "Show me cases where RELIANCE had low PCR and moved up",
            "Based on current FNO data, how much can TCS move tomorrow?",
            "What happens when INFY has high implied volatility?",
            "Find historical patterns for HDFC with high call OI",
            "Show me cases where ICICIBANK had strong momentum",
            "What's the outlook for ITC based on options data?",
            "Find similar market conditions for SBIN",
            "Analyze BHARTIARTL options data for tomorrow"
        ]
        
        for query in test_queries:
            try:
                start_time = time.time()
                
                # Get RAG analysis
                rag_result = self.fno_engine.get_rag_analysis(query, top_k=3)
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Extract key information
                similar_cases = rag_result.get('similar_cases', [])
                ai_analysis = rag_result.get('ai_analysis', '')
                confidence = rag_result.get('confidence', 0.0)
                
                # Count similar cases
                num_similar_cases = len(similar_cases)
                
                # Extract average return from similar cases
                avg_return = 0.0
                if similar_cases:
                    returns = []
                    for case in similar_cases:
                        if 'next_day_move' in case:
                            try:
                                returns.append(float(case['next_day_move']))
                            except:
                                pass
                    if returns:
                        avg_return = np.mean(returns)
                
                result = {
                    'test_type': 'RAG_Analysis',
                    'query': query[:100],  # Truncate long queries
                    'num_similar_cases': num_similar_cases,
                    'avg_return': round(avg_return, 2),
                    'confidence': round(confidence, 2),
                    'ai_analysis_length': len(ai_analysis),
                    'response_time_ms': round(response_time, 2),
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat()
                }
                
                rag_results.append(result)
                logger.info(f"✅ RAG Query: {query[:50]}... - {num_similar_cases} cases, {avg_return:.2f}% avg")
                
            except Exception as e:
                error_result = {
                    'test_type': 'RAG_Analysis',
                    'query': query[:100],
                    'num_similar_cases': 0,
                    'avg_return': 0.0,
                    'confidence': 0.0,
                    'ai_analysis_length': 0,
                    'response_time_ms': 0.0,
                    'status': 'ERROR',
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                rag_results.append(error_result)
                logger.error(f"❌ RAG Query: {query[:50]}... - {e}")
        
        return rag_results
    
    def test_chat_interface(self) -> List[Dict]:
        """Test chat interface with natural language queries."""
        logger.info("🧪 Testing Chat Interface...")
        
        chat_results = []
        chat_queries = [
            "Hello, how are you?",
            "What's the market outlook for today?",
            "Tell me about NIFTY options",
            "What's the PCR for BANKNIFTY?",
            "Show me some trading opportunities",
            "What's the probability of RELIANCE moving up?",
            "Explain implied volatility",
            "What are the best stocks to watch?",
            "How does the RAG system work?",
            "Give me a market summary"
        ]
        
        for query in chat_queries:
            try:
                start_time = time.time()
                
                # Get chat response
                response = self.fno_engine.chat(query)
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                result = {
                    'test_type': 'Chat_Interface',
                    'query': query[:100],
                    'response_length': len(response),
                    'response_time_ms': round(response_time, 2),
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat()
                }
                
                chat_results.append(result)
                logger.info(f"✅ Chat: {query[:30]}... - {len(response)} chars, {response_time:.0f}ms")
                
            except Exception as e:
                error_result = {
                    'test_type': 'Chat_Interface',
                    'query': query[:100],
                    'response_length': 0,
                    'response_time_ms': 0.0,
                    'status': 'ERROR',
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                chat_results.append(error_result)
                logger.error(f"❌ Chat: {query[:30]}... - {e}")
        
        return chat_results
    
    def test_vector_store_performance(self) -> List[Dict]:
        """Test vector store performance metrics."""
        logger.info("🧪 Testing Vector Store Performance...")
        
        performance_results = []
        
        try:
            # Test search performance
            test_queries = [
                "NIFTY high PCR",
                "BANKNIFTY low IV",
                "RELIANCE strong momentum",
                "TCS options data",
                "INFY technical analysis"
            ]
            
            for query in test_queries:
                start_time = time.time()
                
                # Perform vector search
                results = self.vector_store.search_similar_snapshots(query, top_k=5)
                
                end_time = time.time()
                search_time = (end_time - start_time) * 1000  # Convert to ms
                
                result = {
                    'test_type': 'Vector_Store_Performance',
                    'query': query,
                    'num_results': len(results),
                    'search_time_ms': round(search_time, 2),
                    'avg_similarity': round(np.mean([r.get('similarity', 0) for r in results]), 3) if results else 0.0,
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat()
                }
                
                performance_results.append(result)
                logger.info(f"✅ Vector Search: {query} - {len(results)} results, {search_time:.1f}ms")
                
        except Exception as e:
            error_result = {
                'test_type': 'Vector_Store_Performance',
                'query': 'N/A',
                'num_results': 0,
                'search_time_ms': 0.0,
                'avg_similarity': 0.0,
                'status': 'ERROR',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            performance_results.append(error_result)
            logger.error(f"❌ Vector Store Performance: {e}")
        
        return performance_results
    
    def run_comprehensive_test(self) -> pd.DataFrame:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting Comprehensive ML + Enhanced RAG Test")
        logger.info("=" * 60)
        
        # Run all test types
        all_results = []
        
        # Test ML Predictions
        ml_results = self.test_ml_predictions()
        all_results.extend(ml_results)
        
        # Test RAG Analysis
        rag_results = self.test_rag_analysis()
        all_results.extend(rag_results)
        
        # Test Chat Interface
        chat_results = self.test_chat_interface()
        all_results.extend(chat_results)
        
        # Test Vector Store Performance
        performance_results = self.test_vector_store_performance()
        all_results.extend(performance_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Add summary statistics
        self.add_summary_statistics(df)
        
        return df
    
    def add_summary_statistics(self, df: pd.DataFrame) -> None:
        """Add summary statistics to the results."""
        logger.info("📊 Generating Summary Statistics...")
        
        # ML Prediction Summary
        ml_df = df[df['test_type'] == 'ML_Prediction']
        if not ml_df.empty:
            success_rate = (ml_df['status'] == 'SUCCESS').mean() * 100
            avg_response_time = ml_df[ml_df['status'] == 'SUCCESS']['response_time_ms'].mean()
            logger.info(f"📈 ML Predictions: {success_rate:.1f}% success rate, {avg_response_time:.1f}ms avg response")
        
        # RAG Analysis Summary
        rag_df = df[df['test_type'] == 'RAG_Analysis']
        if not rag_df.empty:
            success_rate = (rag_df['status'] == 'SUCCESS').mean() * 100
            avg_response_time = rag_df[rag_df['status'] == 'SUCCESS']['response_time_ms'].mean()
            avg_cases = rag_df[rag_df['status'] == 'SUCCESS']['num_similar_cases'].mean()
            logger.info(f"🔍 RAG Analysis: {success_rate:.1f}% success rate, {avg_response_time:.1f}ms avg response, {avg_cases:.1f} avg cases")
        
        # Chat Interface Summary
        chat_df = df[df['test_type'] == 'Chat_Interface']
        if not chat_df.empty:
            success_rate = (chat_df['status'] == 'SUCCESS').mean() * 100
            avg_response_time = chat_df[chat_df['status'] == 'SUCCESS']['response_time_ms'].mean()
            logger.info(f"💬 Chat Interface: {success_rate:.1f}% success rate, {avg_response_time:.1f}ms avg response")
        
        # Vector Store Performance Summary
        perf_df = df[df['test_type'] == 'Vector_Store_Performance']
        if not perf_df.empty:
            success_rate = (perf_df['status'] == 'SUCCESS').mean() * 100
            avg_search_time = perf_df[perf_df['status'] == 'SUCCESS']['search_time_ms'].mean()
            logger.info(f"⚡ Vector Store: {success_rate:.1f}% success rate, {avg_search_time:.1f}ms avg search time")
    
    def save_results_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save results to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_enhanced_rag_test_results_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Also save summary to separate file
        summary_filename = filename.replace('.csv', '_summary.csv')
        summary_stats = self.generate_summary_stats(df)
        summary_stats.to_csv(summary_filename, index=False)
        
        logger.info(f"💾 Results saved to: {filename}")
        logger.info(f"📊 Summary saved to: {summary_filename}")
        
        return filename
    
    def generate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics."""
        summary_data = []
        
        for test_type in df['test_type'].unique():
            type_df = df[df['test_type'] == test_type]
            
            # Success rate
            success_rate = (type_df['status'] == 'SUCCESS').mean() * 100
            
            # Average response time (for successful tests)
            successful_tests = type_df[type_df['status'] == 'SUCCESS']
            avg_response_time = successful_tests['response_time_ms'].mean() if not successful_tests.empty else 0
            
            # Test-specific metrics
            if test_type == 'ML_Prediction':
                avg_probability = successful_tests['probability'].mean() if not successful_tests.empty else 0
                avg_confidence = successful_tests['confidence'].mean() if not successful_tests.empty else 0
                summary_data.append({
                    'test_type': test_type,
                    'success_rate_percent': round(success_rate, 2),
                    'avg_response_time_ms': round(avg_response_time, 2),
                    'avg_probability': round(avg_probability, 3),
                    'avg_confidence': round(avg_confidence, 3),
                    'total_tests': len(type_df)
                })
            
            elif test_type == 'RAG_Analysis':
                avg_cases = successful_tests['num_similar_cases'].mean() if not successful_tests.empty else 0
                avg_return = successful_tests['avg_return'].mean() if not successful_tests.empty else 0
                summary_data.append({
                    'test_type': test_type,
                    'success_rate_percent': round(success_rate, 2),
                    'avg_response_time_ms': round(avg_response_time, 2),
                    'avg_similar_cases': round(avg_cases, 1),
                    'avg_return_percent': round(avg_return, 2),
                    'total_tests': len(type_df)
                })
            
            elif test_type == 'Chat_Interface':
                avg_response_length = successful_tests['response_length'].mean() if not successful_tests.empty else 0
                summary_data.append({
                    'test_type': test_type,
                    'success_rate_percent': round(success_rate, 2),
                    'avg_response_time_ms': round(avg_response_time, 2),
                    'avg_response_length': round(avg_response_length, 0),
                    'total_tests': len(type_df)
                })
            
            elif test_type == 'Vector_Store_Performance':
                avg_search_time = successful_tests['search_time_ms'].mean() if not successful_tests.empty else 0
                avg_similarity = successful_tests['avg_similarity'].mean() if not successful_tests.empty else 0
                summary_data.append({
                    'test_type': test_type,
                    'success_rate_percent': round(success_rate, 2),
                    'avg_search_time_ms': round(avg_search_time, 2),
                    'avg_similarity': round(avg_similarity, 3),
                    'total_tests': len(type_df)
                })
        
        return pd.DataFrame(summary_data)

def main():
    """Main test execution function."""
    print("🧪 ML + Enhanced RAG System Test")
    print("=" * 50)
    
    try:
        # Initialize tester
        tester = MLEnhancedRAGTester()
        
        # Run comprehensive test
        results_df = tester.run_comprehensive_test()
        
        # Save results to CSV
        csv_filename = tester.save_results_to_csv(results_df)
        
        # Display summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        # Show results by test type
        for test_type in results_df['test_type'].unique():
            type_df = results_df[results_df['test_type'] == test_type]
            success_count = (type_df['status'] == 'SUCCESS').sum()
            total_count = len(type_df)
            success_rate = (success_count / total_count) * 100
            
            print(f"\n{test_type}:")
            print(f"  ✅ Success: {success_count}/{total_count} ({success_rate:.1f}%)")
            
            if success_count > 0:
                successful_tests = type_df[type_df['status'] == 'SUCCESS']
                avg_time = successful_tests['response_time_ms'].mean()
                print(f"  ⏱️  Avg Response Time: {avg_time:.1f}ms")
        
        print(f"\n💾 Detailed results saved to: {csv_filename}")
        print(f"📊 Summary saved to: {csv_filename.replace('.csv', '_summary.csv')}")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
