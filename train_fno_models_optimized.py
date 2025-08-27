#!/usr/bin/env python3
"""
Optimized FNO ML Models Training
Fast training with reduced data and simpler models for quick testing.
"""

import sys
import os
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, HorizonType
from src.fno_rag.core.data_processor import FNODataProcessor
from loguru import logger


def train_fno_models_optimized():
    """Train FNO ML models with optimizations for speed."""
    
    print("🚀 Optimized FNO ML Models Training")
    print("=" * 50)
    
    try:
        # Initialize FNO engine
        print("1. Initializing FNO RAG System...")
        start_time = time.time()
        fno_engine = FNOEngine()
        init_time = time.time() - start_time
        print(f"   ✅ Initialized in {init_time:.2f} seconds")
        
        # Get data processor
        data_processor = fno_engine.data_processor
        
        # Get historical data for training (reduced to 1 month for speed)
        print("\n2. Loading historical data for training...")
        data_start = time.time()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"   Loading data from {start_date} to {end_date}")
        print("   ⏳ This may take a few minutes...")
        
        df = data_processor.get_fno_data(start_date=start_date, end_date=end_date)
        
        if df.empty:
            print("   ❌ No data available for training")
            return False
        
        data_time = time.time() - data_start
        print(f"   ✅ Loaded {len(df)} records in {data_time:.2f} seconds")
        
        # Sample data for faster training (take 20% of data)
        if len(df) > 10000:
            print(f"   📊 Sampling 20% of data for faster training...")
            df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
            print(f"   ✅ Sampled {len(df)} records")
        
        # Calculate technical indicators and labels
        print("\n3. Calculating technical indicators and labels...")
        indicators_start = time.time()
        df = data_processor.calculate_technical_indicators(df)
        df = data_processor.create_labels(df)
        indicators_time = time.time() - indicators_start
        
        print(f"   ✅ Processed data with {len(df)} samples in {indicators_time:.2f} seconds")
        
        # Train models for all horizons at once
        print(f"\n4. Training models for all horizons...")
        training_start = time.time()
        
        try:
            # Train all models using the processed DataFrame
            results = fno_engine.train_models(df=df)
            training_time = time.time() - training_start
            
            print(f"   ✅ Models trained successfully in {training_time:.2f} seconds")
            print(f"   📊 Results: {list(results.keys())}")
            
            # Show model performance
            for horizon, result in results.items():
                if 'accuracy' in result:
                    print(f"   📈 {horizon}: Accuracy = {result['accuracy']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed to train models: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n5. Model training completed!")
        
        # Quick test of trained models
        print("\n6. Quick testing of trained models...")
        test_symbols = ['NIFTY', 'BANKNIFTY']  # Reduced test symbols
        horizons = [HorizonType.DAILY]  # Test only daily for speed
        
        for symbol in test_symbols:
            for horizon in horizons:
                try:
                    result = fno_engine.predict_probability(symbol, horizon)
                    print(f"   ✅ {symbol} ({horizon.value}): Up={result.up_probability:.3f}, Down={result.down_probability:.3f}")
                except Exception as e:
                    print(f"   ❌ {symbol} ({horizon.value}): {e}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Optimized FNO ML Models training completed in {total_time:.2f} seconds!")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Optimized FNO ML Models Training...")
    success = train_fno_models_optimized()
    
    if success:
        print(f"\n🎉 Training completed successfully!")
    else:
        print(f"\n❌ Training failed!")
