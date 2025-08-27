#!/usr/bin/env python3
"""
FNO RAG Model Training Script
Trains ML models for FNO probability prediction across multiple horizons.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.fno_rag import FNOEngine
from src.fno_rag.models.data_models import HorizonType
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta


def main():
    """Main training function."""
    try:
        logger.info("🚀 Starting FNO RAG Model Training...")
        
        # Initialize FNO Engine
        fno_engine = FNOEngine()
        
        # Check if system is initialized
        if not fno_engine.initialized:
            logger.error("❌ FNO Engine failed to initialize")
            return False
        
        logger.info("✅ FNO Engine initialized successfully")
        
        # Get training data info
        data_processor = fno_engine.data_processor
        test_data = data_processor.get_fno_data()
        
        if test_data is None or len(test_data) == 0:
            logger.error("❌ No training data available")
            return False
        
        logger.info(f"📊 Training data available: {len(test_data)} records")
        logger.info(f"📊 Data columns: {list(test_data.columns)}")
        
        # Check for required columns
        required_columns = ['symbol', 'date', 'close_price', 'volume']
        missing_columns = [col for col in required_columns if col not in test_data.columns]
        
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Get date range
        if 'date' in test_data.columns:
            test_data['date'] = pd.to_datetime(test_data['date'])
            start_date = test_data['date'].min()
            end_date = test_data['date'].max()
            logger.info(f"📅 Data date range: {start_date} to {end_date}")
        
        # Train models
        logger.info("🎯 Starting model training...")
        
        training_results = fno_engine.train_models(
            symbols=None,  # Use all symbols
            start_date=None,  # Use all available data
            end_date=None,
            df=test_data
        )
        
        if not training_results:
            logger.error("❌ Model training failed")
            return False
        
        # Display training results
        logger.info("📈 Training Results:")
        for horizon, result in training_results.items():
            logger.info(f"\n🎯 {horizon.upper()} Model:")
            logger.info(f"   Model Type: {result.get('model_type', 'N/A')}")
            logger.info(f"   Accuracy: {result.get('accuracy', 0):.3f}")
            logger.info(f"   CV Mean: {result.get('cv_mean', 0):.3f} ± {result.get('cv_std', 0):.3f}")
            logger.info(f"   Training Samples: {result.get('training_samples', 0)}")
            logger.info(f"   Test Samples: {result.get('test_samples', 0)}")
            
            # Show top features
            feature_importance = result.get('feature_importance', {})
            if feature_importance:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"   Top Features:")
                for feature, importance in top_features:
                    logger.info(f"     {feature}: {importance:.3f}")
        
        # Test predictions
        logger.info("\n🧪 Testing predictions...")
        
        # Get unique symbols
        symbols = test_data['symbol'].unique()[:5]  # Test with first 5 symbols
        
        for symbol in symbols:
            logger.info(f"\n📊 Testing {symbol}:")
            
            for horizon in HorizonType:
                try:
                    result = fno_engine.predict_probability(symbol, horizon)
                    logger.info(f"   {horizon.value}: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}")
                except Exception as e:
                    logger.warning(f"   {horizon.value}: Failed - {e}")
        
        # Get system status
        logger.info("\n📊 System Status:")
        status = fno_engine.get_system_status()
        
        for component, component_status in status.items():
            logger.info(f"   {component}: {'✅' if component_status else '❌'}")
        
        # Get model status
        logger.info("\n🤖 Model Status:")
        model_status = fno_engine.ml_models.get_model_status()
        
        for horizon, status in model_status.items():
            trained = status.get('trained', False)
            model_type = status.get('model_type', 'unknown')
            trained_at = status.get('trained_at', 'unknown')
            
            logger.info(f"   {horizon}: {'✅' if trained else '❌'} ({model_type}) - {trained_at}")
        
        logger.info("\n🎉 FNO RAG Model Training Completed Successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False


def train_specific_horizon(horizon: HorizonType):
    """Train a specific horizon model."""
    try:
        logger.info(f"🎯 Training {horizon.value} model...")
        
        fno_engine = FNOEngine()
        
        if not fno_engine.initialized:
            logger.error("❌ FNO Engine failed to initialize")
            return False
        
        # Get data
        data_processor = fno_engine.data_processor
        test_data = data_processor.get_fno_data()
        
        if test_data is None or len(test_data) == 0:
            logger.error("❌ No training data available")
            return False
        
        # Train specific model
        result = fno_engine.ml_models.retrain_model(horizon, df=test_data)
        
        logger.info(f"✅ {horizon.value} model trained successfully")
        logger.info(f"   Accuracy: {result.get('accuracy', 0):.3f}")
        logger.info(f"   CV Mean: {result.get('cv_mean', 0):.3f} ± {result.get('cv_std', 0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to train {horizon.value} model: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FNO RAG Models")
    parser.add_argument("--horizon", choices=["daily", "weekly", "monthly"], 
                       help="Train specific horizon model")
    
    args = parser.parse_args()
    
    if args.horizon:
        # Train specific horizon
        horizon_map = {
            "daily": HorizonType.DAILY,
            "weekly": HorizonType.WEEKLY,
            "monthly": HorizonType.MONTHLY
        }
        success = train_specific_horizon(horizon_map[args.horizon])
    else:
        # Train all models
        success = main()
    
    if success:
        logger.info("🎉 Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)
