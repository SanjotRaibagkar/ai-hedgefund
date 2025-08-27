#!/usr/bin/env python3
"""
Chunked FNO Vector Store Builder
Build the RAG vector store using divide-and-conquer approach to avoid memory issues.
"""

import sys
import os
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.core.data_processor import FNODataProcessor
from src.fno_rag.utils.embedding_utils import EmbeddingUtils
from src.fno_rag.rag.vector_store import FNOVectorStore
from loguru import logger

def build_vector_store_chunked():
    """Build the FNO vector store using chunked processing."""
    
    print("🏗️ Building FNO Vector Store (Chunked Approach)")
    print("=" * 60)
    
    try:
        # Initialize components
        print("1. Initializing components...")
        data_processor = FNODataProcessor()
        embedding_utils = EmbeddingUtils()
        vector_store = FNOVectorStore()
        
        # Get all symbols
        print("2. Getting available symbols...")
        symbols = data_processor.get_available_symbols()
        print(f"   📊 Found {len(symbols)} symbols")
        
        # Process in chunks of 10 symbols
        chunk_size = 10
        total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
        
        print(f"3. Processing {len(symbols)} symbols in {total_chunks} chunks...")
        
        all_market_conditions = []
        chunk_count = 0
        
        for i in range(0, len(symbols), chunk_size):
            chunk_count += 1
            chunk_symbols = symbols[i:i + chunk_size]
            
            print(f"   📦 Processing chunk {chunk_count}/{total_chunks}: {len(chunk_symbols)} symbols")
            
            try:
                # Get data for this chunk
                df_chunk = data_processor.get_fno_data(symbols=chunk_symbols, limit=1000)
                
                if df_chunk.empty:
                    print(f"      ⚠️ No data for chunk {chunk_count}")
                    continue
                
                # Process each symbol in the chunk
                for symbol in chunk_symbols:
                    symbol_data = df_chunk[df_chunk['TckrSymb'] == symbol]
                    
                    if symbol_data.empty:
                        continue
                    
                    # Create market conditions for this symbol
                    conditions = create_market_conditions(symbol_data, symbol)
                    all_market_conditions.extend(conditions)
                    
                    if len(all_market_conditions) % 100 == 0:
                        print(f"      ✅ Processed {len(all_market_conditions)} conditions so far")
                
                # Clear chunk data to free memory
                del df_chunk
                
            except Exception as e:
                print(f"      ❌ Error processing chunk {chunk_count}: {e}")
                continue
        
        print(f"4. Creating vector store with {len(all_market_conditions)} conditions...")
        
        # Create embeddings in batches
        batch_size = 100
        embeddings = []
        texts = []
        
        for i in range(0, len(all_market_conditions), batch_size):
            batch = all_market_conditions[i:i + batch_size]
            batch_texts = [condition['text'] for condition in batch]
            
            # Create embeddings for this batch
            batch_embeddings = embedding_utils.create_embeddings(batch_texts)
            embeddings.extend(batch_embeddings)
            texts.extend(batch_texts)
            
            print(f"   🔄 Processed batch {i//batch_size + 1}/{(len(all_market_conditions) + batch_size - 1)//batch_size}")
        
        # Build vector store using the existing method
        print("5. Building vector store...")
        vector_store.build_vector_store()
        
        print("6. Vector store built successfully!")
        
        print(f"✅ Vector store built successfully!")
        print(f"   📊 Total conditions: {len(all_market_conditions)}")
        print(f"   🗂️ Index size: {len(embeddings)} vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building vector store: {e}")
        logger.error(f"Vector store build failed: {e}")
        return False

def create_market_conditions(df: pd.DataFrame, symbol: str) -> list:
    """Create market conditions from symbol data."""
    conditions = []
    
    try:
        # Group by date to get daily conditions
        df['date'] = pd.to_datetime(df['date'])
        daily_data = df.groupby(df['date'].dt.date).agg({
            'open_price': 'first',
            'high_price': 'max',
            'low_price': 'min',
            'close_price': 'last',
            'volume': 'sum',
            'open_interest': 'last'
        }).reset_index()
        
        # Calculate daily returns
        daily_data['daily_return'] = daily_data['close_price'].pct_change()
        daily_data['volume_change'] = daily_data['volume'].pct_change()
        daily_data['oi_change'] = daily_data['open_interest'].pct_change()
        
        # Create conditions for each day
        for idx, row in daily_data.iterrows():
            if pd.isna(row['daily_return']):
                continue
                
            # Determine market condition
            if row['daily_return'] >= 0.03:
                condition = "bullish"
                direction = "up"
            elif row['daily_return'] <= -0.03:
                condition = "bearish"
                direction = "down"
            else:
                condition = "neutral"
                direction = "sideways"
            
            # Create descriptive text
            text = f"{symbol} {condition} market on {row['date']}: {direction} {abs(row['daily_return'])*100:.1f}% return, volume change {row['volume_change']*100:.1f}%, OI change {row['oi_change']*100:.1f}%"
            
            conditions.append({
                'symbol': symbol,
                'date': row['date'],
                'condition': condition,
                'direction': direction,
                'daily_return': row['daily_return'],
                'volume_change': row['volume_change'],
                'oi_change': row['oi_change'],
                'text': text,
                'features': {
                    'daily_return': row['daily_return'],
                    'volume_change': row['volume_change'],
                    'oi_change': row['oi_change'],
                    'high_low_ratio': (row['high_price'] - row['low_price']) / row['close_price']
                }
            })
    
    except Exception as e:
        print(f"      ⚠️ Error creating conditions for {symbol}: {e}")
    
    return conditions

if __name__ == "__main__":
    build_vector_store_chunked()
