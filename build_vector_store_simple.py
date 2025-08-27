#!/usr/bin/env python3
"""
Simple FNO Vector Store Builder
Build the RAG vector store with limited data to avoid memory issues.
"""

import sys
import os
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.rag.vector_store import FNOVectorStore
from loguru import logger

def build_vector_store_simple():
    """Build the FNO vector store with limited data."""
    
    print("🏗️ Building FNO Vector Store (Simple Approach)")
    print("=" * 50)
    
    try:
        # Initialize vector store
        print("1. Initializing vector store...")
        vector_store = FNOVectorStore()
        print("   ✅ Vector store initialized")
        
        # Build with limited data (top 10 symbols, last 30 days)
        print("2. Building vector store with limited data...")
        print("   📊 Using top 10 symbols, last 30 days")
        
        # Top FNO symbols
        symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN']
        
        # Date range (last 30 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"   📅 Date range: {start_date} to {end_date}")
        
        # Build vector store
        stats = vector_store.build_vector_store(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=100  # Small batch size to avoid memory issues
        )
        
        print("3. Vector store statistics:")
        for key, value in stats.items():
            print(f"   📊 {key}: {value}")
        
        print("✅ Vector store built successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error building vector store: {e}")
        logger.error(f"Vector store build failed: {e}")
        return False

if __name__ == "__main__":
    build_vector_store_simple()
