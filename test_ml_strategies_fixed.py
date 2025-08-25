#!/usr/bin/env python3
"""
Test ML Strategies Module - Fixed Version
Demonstrates the functionality with available data from the database.
"""

import sys
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from src.ml.ml_strategies import MLEnhancedEODStrategy

def check_available_data():
    """Check what data is available in the database."""
    print("🔍 CHECKING AVAILABLE DATA")
    print("=" * 50)
    
    try:
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Check available symbols
        symbols = conn.execute('SELECT DISTINCT symbol FROM price_data ORDER BY symbol LIMIT 10').fetchall()
        print(f"Available symbols (first 10): {[s[0] for s in symbols]}")
        
        # Check date range
        date_range = conn.execute('SELECT MIN(date), MAX(date) FROM price_data').fetchall()
        print(f"Date range: {date_range[0]}")
        
        # Check data count for a specific symbol
        sample_symbol = symbols[0][0] if symbols else 'ALPEXSOLAR'
        count = conn.execute('SELECT COUNT(*) FROM price_data WHERE symbol = ?', [sample_symbol]).fetchall()
        print(f"Data points for {sample_symbol}: {count[0][0]}")
        
        # Get sample data
        sample_data = conn.execute('''
            SELECT symbol, date, open_price, high_price, low_price, close_price, volume 
            FROM price_data 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 5
        ''', [sample_symbol]).fetchall()
        
        print(f"\nSample data for {sample_symbol}:")
        for row in sample_data:
            print(f"  {row[0]} | {row[1]} | O:{row[2]:.2f} H:{row[3]:.2f} L:{row[4]:.2f} C:{row[5]:.2f} V:{row[6]}")
        
        conn.close()
        return sample_symbol, date_range[0]
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return None, None

def test_ml_strategies_with_available_data():
    """Test the ML Strategies module with available data."""
    print("\n🚀 TESTING ML STRATEGIES MODULE")
    print("=" * 80)
    
    # First check available data
    symbol, date_range = check_available_data()
    if not symbol or not date_range:
        print("❌ No data available for testing")
        return
    
    start_date, end_date = date_range
    
    try:
        # Initialize the strategy
        print(f"\n📊 1. Initializing ML-Enhanced EOD Strategy...")
        strategy = MLEnhancedEODStrategy()
        print("✅ Strategy initialized successfully")
        
        # Get strategy summary
        print(f"\n📋 2. Strategy Configuration:")
        summary = strategy.get_strategy_summary()
        print(f"   • Strategy Type: {summary['strategy_type']}")
        print(f"   • Model Type: {summary['model_type']}")
        print(f"   • Model Trained: {summary['model_trained']}")
        print(f"   • Momentum Weight: {summary['strategy_config']['momentum_weight']}")
        print(f"   • ML Weight: {summary['strategy_config']['ml_weight']}")
        print(f"   • Confidence Threshold: {summary['strategy_config']['confidence_threshold']}")
        
        print(f"\n📅 3. Testing with Date Range: {start_date} to {end_date}")
        print(f"🎯 4. Testing with Stock: {symbol}")
        
        # Train the model
        print(f"\n🤖 5. Training ML Model for {symbol}...")
        training_results = strategy.train_model(
            ticker=symbol,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        
        if 'error' in training_results:
            print(f"❌ Training failed: {training_results['error']}")
            return
        
        print("✅ Model training completed")
        print(f"   • Model Type: {training_results['model_type']}")
        print(f"   • Test R² Score: {training_results['model_performance']['test_r2']:.4f}")
        print(f"   • Test RMSE: {training_results['model_performance']['test_rmse']:.4f}")
        print(f"   • Training Samples: {training_results['model_performance']['training_samples']}")
        print(f"   • Test Samples: {training_results['model_performance']['test_samples']}")
        
        # Make predictions
        print(f"\n🔮 6. Making Predictions for {symbol}...")
        prediction_results = strategy.predict_returns(
            ticker=symbol,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        
        if 'error' in prediction_results:
            print(f"❌ Prediction failed: {prediction_results['error']}")
            return
        
        print("✅ Predictions completed")
        print(f"   • Latest Prediction: {prediction_results['latest_prediction']:.4f}")
        print(f"   • Prediction Confidence: {prediction_results['prediction_confidence']:.4f}")
        
        if not prediction_results['predictions'].empty:
            print(f"   • Total Predictions: {len(prediction_results['predictions'])}")
            print(f"   • Prediction Range: {prediction_results['predictions']['predicted_return'].min():.4f} to {prediction_results['predictions']['predicted_return'].max():.4f}")
        
        # Get complete analysis
        print(f"\n📈 7. Performing Complete Analysis for {symbol}...")
        analysis_results = strategy.analyze_stock(
            ticker=symbol,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        
        if 'error' in analysis_results:
            print(f"❌ Analysis failed: {analysis_results['error']}")
            return
        
        print("✅ Analysis completed")
        
        # Display combined signals
        combined_signals = analysis_results['combined_signals']
        print(f"\n📊 8. Combined Signals:")
        print(f"   • Momentum Signal: {combined_signals['momentum_signal']:.4f}")
        print(f"   • ML Signal: {combined_signals['ml_signal']:.4f}")
        print(f"   • Combined Signal: {combined_signals['combined_signal']:.4f}")
        print(f"   • Combined Confidence: {combined_signals['combined_confidence']:.4f}")
        
        # Display final recommendation
        final_rec = analysis_results['final_recommendation']
        print(f"\n🎯 9. Final Trading Recommendation:")
        print(f"   • Action: {final_rec['action']}")
        print(f"   • Strength: {final_rec['strength']}")
        print(f"   • Confidence: {final_rec['confidence']:.4f}")
        print(f"   • Position Size: {final_rec['position_size']:.4f}")
        print(f"   • Reasoning: {final_rec['reasoning']}")
        
        # Display feature importance if available
        if 'feature_importance' in training_results and training_results['feature_importance']:
            print(f"\n🔍 10. Top Feature Importance:")
            feature_importance = training_results['feature_importance']
            # Sort by importance and show top 5
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                print(f"   • {feature}: {importance:.4f}")
        
        print(f"\n✅ ML Strategies Module Test Completed Successfully!")
        print(f"📅 Analysis Timestamp: {analysis_results['analysis_timestamp']}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_feature_engineering():
    """Test feature engineering separately."""
    print("\n" + "=" * 80)
    print("🔧 TESTING FEATURE ENGINEERING")
    print("=" * 80)
    
    try:
        from src.ml.feature_engineering import FeatureEngineer
        
        # Get sample data
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        sample_symbol = conn.execute('SELECT DISTINCT symbol FROM price_data LIMIT 1').fetchall()[0][0]
        
        # Get recent data
        data = conn.execute('''
            SELECT date, open_price, high_price, low_price, close_price, volume
            FROM price_data 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 100
        ''', [sample_symbol]).fetchdf()
        
        conn.close()
        
        if data.empty:
            print("❌ No data available for feature engineering test")
            return
        
        print(f"📊 Testing feature engineering with {sample_symbol}")
        print(f"   • Data points: {len(data)}")
        print(f"   • Date range: {data['date'].min()} to {data['date'].max()}")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Create features
        features, target = feature_engineer.create_features(
            ticker=sample_symbol,
            start_date=str(data['date'].min()),
            end_date=str(data['date'].max()),
            target_horizon=5
        )
        
        print(f"✅ Feature engineering completed")
        print(f"   • Features created: {features.shape[1] if not features.empty else 0}")
        print(f"   • Target samples: {len(target) if not target.empty else 0}")
        
        if not features.empty:
            print(f"   • Feature columns: {list(features.columns)[:10]}...")
            print(f"   • Sample feature values:")
            for col in features.columns[:5]:
                if col in features.columns:
                    print(f"     - {col}: {features[col].iloc[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_strategies_with_available_data()
    test_feature_engineering()
