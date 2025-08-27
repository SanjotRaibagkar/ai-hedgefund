#!/usr/bin/env python3
"""
Simple Model Testing
Test the trained FNO ML models directly.
"""

import sys
import os
import pickle
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.models.data_models import HorizonType


def test_models():
    """Test the trained models directly."""
    
    print("🧪 Simple Model Testing")
    print("=" * 50)
    
    try:
        # Check if models exist
        models_dir = Path("models/fno_ml")
        
        if not models_dir.exists():
            print("❌ Models directory not found")
            return False
        
        print(f"📁 Models directory: {models_dir}")
        
        # List model files
        model_files = list(models_dir.glob("*_model.pkl"))
        print(f"📊 Found {len(model_files)} model files:")
        
        for model_file in model_files:
            print(f"   📄 {model_file.name}")
        
        # Test loading each model
        for horizon in HorizonType:
            model_file = models_dir / f"{horizon.value}_model.pkl"
            scaler_file = models_dir / f"{horizon.value}_scaler.pkl"
            
            if model_file.exists() and scaler_file.exists():
                print(f"\n🔍 Testing {horizon.value} model...")
                
                try:
                    # Load model
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Load scaler
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    print(f"   ✅ Model loaded successfully")
                    print(f"   📊 Model type: {type(model).__name__}")
                    
                    # Test with dummy features (33 features as seen in training)
                    dummy_features = [0.0] * 33
                    
                    # Scale features
                    X_scaled = scaler.transform([dummy_features])
                    
                    # Get prediction
                    proba = model.predict_proba(X_scaled)[0]
                    
                    print(f"   🎯 Prediction probabilities: {proba}")
                    print(f"   📈 Up probability: {proba[2]:.3f}")
                    print(f"   📉 Down probability: {proba[0]:.3f}")
                    print(f"   ➡️ Neutral probability: {proba[1]:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to test model: {e}")
            else:
                print(f"\n⚠️ {horizon.value} model files not found")
        
        print(f"\n🎉 Model testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Starting Simple Model Testing...")
    success = test_models()
    
    if success:
        print(f"\n🎉 Testing completed successfully!")
    else:
        print(f"\n❌ Testing failed!")
