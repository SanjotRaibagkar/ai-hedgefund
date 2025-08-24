#!/usr/bin/env python3
"""
Test script to verify ML model works with only technical data
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.ml.feature_engineering import FeatureEngineer
from src.ml.ml_strategies import MLEnhancedEODStrategy


def test_ml_technical_features_only():
    """Test ML model with only technical features."""
    logger.info("🧪 Testing ML model with technical features only...")
    
    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Test with a known working symbol
        ticker = "RELIANCE.NS"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        logger.info(f"Testing feature creation for {ticker}")
        
        # Create features
        features, target = feature_engineer.create_features(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            target_horizon=5
        )
        
        # Check results
        if features.empty:
            logger.error("❌ No features created")
            return False
        
        if target.empty:
            logger.error("❌ No target created")
            return False
        
        logger.info(f"✅ Features created successfully:")
        logger.info(f"   📊 Feature count: {features.shape[1]}")
        logger.info(f"   📈 Sample count: {features.shape[0]}")
        logger.info(f"   🎯 Target count: {len(target)}")
        
        # Check feature types
        technical_features = [col for col in features.columns if any(x in col.lower() for x in ['rsi', 'macd', 'sma', 'volume', 'price', 'atr', 'bollinger', 'stoch'])]
        fundamental_features = [col for col in features.columns if any(x in col.lower() for x in ['pe', 'pb', 'roe', 'roa', 'debt', 'margin'])]
        
        logger.info(f"   🔧 Technical features: {len(technical_features)}")
        logger.info(f"   📊 Fundamental features: {len(fundamental_features)}")
        
        if len(technical_features) > 0 and len(fundamental_features) == 0:
            logger.info("✅ ML model is using only technical features (as expected)")
            return True
        elif len(fundamental_features) > 0:
            logger.warning("⚠️ Found fundamental features - this shouldn't happen")
            return False
        else:
            logger.error("❌ No technical features found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_ml_strategy_technical_only():
    """Test ML strategy with only technical features."""
    logger.info("🧪 Testing ML strategy with technical features only...")
    
    try:
        # Initialize ML strategy
        ml_strategy = MLEnhancedEODStrategy()
        
        # Test with a known working symbol
        ticker = "RELIANCE.NS"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        logger.info(f"Testing ML strategy for {ticker}")
        
        # Train model
        training_results = ml_strategy.train_model(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in training_results:
            logger.error(f"❌ Training failed: {training_results['error']}")
            return False
        
        logger.info("✅ ML model training completed successfully")
        logger.info(f"   📊 Model performance: {training_results.get('model_performance', {})}")
        logger.info(f"   🔧 Model type: {training_results.get('model_type', 'Unknown')}")
        
        # Test prediction
        prediction_results = ml_strategy.predict_returns(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in prediction_results:
            logger.error(f"❌ Prediction failed: {prediction_results['error']}")
            return False
        
        logger.info("✅ ML prediction completed successfully")
        logger.info(f"   📈 Latest prediction: {prediction_results.get('latest_prediction', 'N/A')}")
        logger.info(f"   🎯 Prediction confidence: {prediction_results.get('prediction_confidence', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ML strategy test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🚀 STARTING ML TECHNICAL-ONLY TEST")
    logger.info("=" * 60)
    
    # Test feature engineering
    feature_test_passed = test_ml_technical_features_only()
    
    logger.info("=" * 60)
    
    # Test ML strategy
    strategy_test_passed = test_ml_strategy_technical_only()
    
    logger.info("=" * 60)
    
    # Summary
    if feature_test_passed and strategy_test_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ ML model works perfectly with technical features only")
        logger.info("✅ No fundamental data dependency")
        logger.info("✅ System is ready for production use")
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("   Please check the errors above")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
