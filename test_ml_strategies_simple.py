#!/usr/bin/env python3
"""
Simple ML Strategies Test
Demonstrates the functionality using direct database access.
"""

import sys
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def test_ml_strategies_simple():
    """Test ML strategies with direct database access."""
    print("🚀 SIMPLE ML STRATEGIES TEST")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Get a stock with good data
        symbols = conn.execute('''
            SELECT symbol, COUNT(*) as data_points 
            FROM price_data 
            GROUP BY symbol 
            HAVING COUNT(*) > 200 
            ORDER BY COUNT(*) DESC 
            LIMIT 5
        ''').fetchall()
        
        if not symbols:
            print("❌ No stocks with sufficient data found")
            return
        
        test_symbol = symbols[0][0]
        print(f"📊 Testing with: {test_symbol} ({symbols[0][1]} data points)")
        
        # Get price data
        data = conn.execute('''
            SELECT date, open_price, high_price, low_price, close_price, volume
            FROM price_data 
            WHERE symbol = ? 
            ORDER BY date
        ''', [test_symbol]).fetchdf()
        
        conn.close()
        
        if data.empty:
            print("❌ No data found for the symbol")
            return
        
        print(f"✅ Retrieved {len(data)} data points")
        print(f"   • Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"   • Price range: ₹{data['close_price'].min():.2f} to ₹{data['close_price'].max():.2f}")
        
        # Create simple features
        print("\n🔧 Creating Features...")
        features = create_simple_features(data)
        print(f"✅ Created {features.shape[1]} features")
        
        # Create target (next day return)
        target = data['close_price'].pct_change().shift(-1)
        
        # Align features and target properly
        # Remove rows where we don't have target (last row) or features (first few rows due to rolling windows)
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]
        
        print(f"✅ Final dataset: {len(features)} samples, {features.shape[1]} features")
        print(f"   • Target range: {target.min():.4f} to {target.max():.4f}")
        
        if len(features) < 50:
            print("❌ Insufficient data for training")
            return
        
        # Train model
        print("\n🤖 Training ML Model...")
        model, scaler, performance = train_simple_model(features, target)
        
        print("✅ Model training completed")
        print(f"   • R² Score: {performance['r2_score']:.4f}")
        print(f"   • RMSE: {performance['rmse']:.4f}")
        print(f"   • Training samples: {performance['train_samples']}")
        print(f"   • Test samples: {performance['test_samples']}")
        
        # Make predictions
        print("\n🔮 Making Predictions...")
        latest_features = features.iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        prediction = model.predict(latest_features_scaled)[0]
        
        print(f"✅ Latest prediction: {prediction:.4f} ({prediction*100:.2f}%)")
        
        # Generate trading signal
        print("\n📊 Generating Trading Signal...")
        signal = generate_trading_signal(prediction, features.iloc[-1])
        
        print("✅ Trading Signal Generated")
        print(f"   • Signal: {signal['action']}")
        print(f"   • Confidence: {signal['confidence']:.4f}")
        print(f"   • Reasoning: {signal['reasoning']}")
        
        # Show feature importance
        print("\n🔍 Feature Importance (Top 5):")
        feature_importance = dict(zip(features.columns, model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature, importance in sorted_features:
            print(f"   • {feature}: {importance:.4f}")
        
        print(f"\n✅ ML Strategies Test Completed Successfully!")
        print(f"📅 Test completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def create_simple_features(data):
    """Create simple technical features."""
    features = pd.DataFrame(index=data.index)
    
    # Price features
    features['price_change_1d'] = data['close_price'].pct_change(1)
    features['price_change_5d'] = data['close_price'].pct_change(5)
    features['price_change_20d'] = data['close_price'].pct_change(20)
    
    # Moving averages
    features['sma_5'] = data['close_price'].rolling(5).mean()
    features['sma_20'] = data['close_price'].rolling(20).mean()
    features['price_to_sma5'] = data['close_price'] / features['sma_5']
    features['price_to_sma20'] = data['close_price'] / features['sma_20']
    
    # Volume features
    features['volume_sma_5'] = data['volume'].rolling(5).mean()
    features['volume_ratio'] = data['volume'] / features['volume_sma_5']
    
    # Volatility features
    features['volatility_5d'] = data['close_price'].pct_change().rolling(5).std()
    features['volatility_20d'] = data['close_price'].pct_change().rolling(20).std()
    
    # Price position
    features['high_low_ratio'] = data['high_price'] / data['low_price']
    features['close_to_high'] = data['close_price'] / data['high_price']
    features['close_to_low'] = data['close_price'] / data['low_price']
    
    # Momentum features
    features['momentum_5d'] = data['close_price'] / data['close_price'].shift(5) - 1
    features['momentum_20d'] = data['close_price'] / data['close_price'].shift(20) - 1
    
    # RSI-like feature
    delta = data['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    return features

def train_simple_model(features, target):
    """Train a simple Random Forest model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate performance
    performance = {
        'r2_score': r2_score(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return model, scaler, performance

def generate_trading_signal(prediction, latest_features):
    """Generate trading signal based on prediction and features."""
    # Simple signal generation
    if prediction > 0.02:  # 2% positive return
        action = "BUY"
        confidence = min(abs(prediction) * 10, 1.0)
        reasoning = "ML model predicts strong positive returns"
    elif prediction < -0.02:  # 2% negative return
        action = "SELL"
        confidence = min(abs(prediction) * 10, 1.0)
        reasoning = "ML model predicts strong negative returns"
    else:
        action = "HOLD"
        confidence = 0.5
        reasoning = "ML model predicts minimal movement"
    
    # Adjust based on technical indicators
    if latest_features['price_to_sma20'] > 1.1:
        reasoning += "; Stock trading above 20-day SMA"
    elif latest_features['price_to_sma20'] < 0.9:
        reasoning += "; Stock trading below 20-day SMA"
    
    if latest_features['rsi'] > 70:
        reasoning += "; RSI indicates overbought conditions"
    elif latest_features['rsi'] < 30:
        reasoning += "; RSI indicates oversold conditions"
    
    return {
        'action': action,
        'confidence': confidence,
        'reasoning': reasoning,
        'prediction': prediction
    }

if __name__ == "__main__":
    test_ml_strategies_simple()
