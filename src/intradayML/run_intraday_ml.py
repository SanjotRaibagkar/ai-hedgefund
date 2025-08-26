#!/usr/bin/env python3
"""
Intraday ML Runner
Main script to run the intraday ML prediction system.
"""

import sys
import os
sys.path.append('./src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import argparse

from intradayML import (
    IntradayDataCollector,
    IntradayFeatureEngineer,
    IntradayMLTrainer,
    IntradayPredictor,
    IntradayUtils
)


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        "logs/intraday_ml.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
    )


def collect_data_demo():
    """Demonstrate data collection functionality."""
    print("\n" + "="*60)
    print("📊 DATA COLLECTION DEMO")
    print("="*60)
    
    try:
        # Initialize data collector
        collector = IntradayDataCollector()
        
        # Collect data for NIFTY and BANKNIFTY
        index_symbols = ['NIFTY', 'BANKNIFTY']
        
        for index_symbol in index_symbols:
            print(f"\n🔍 Collecting data for {index_symbol}...")
            
            # Collect options chain data
            options_data = collector.collect_options_chain_data(index_symbol)
            print(f"   ✅ Options data: {len(options_data)} records")
            
            # Collect index data
            index_data = collector.collect_index_data(index_symbol)
            print(f"   ✅ Index data: {len(index_data)} records")
        
        # Collect FII/DII data
        print(f"\n💰 Collecting FII/DII data...")
        fii_dii_data = collector.collect_fii_dii_data()
        print(f"   ✅ FII/DII data: {len(fii_dii_data)} records")
        
        # Collect VIX data
        print(f"\n📊 Collecting VIX data...")
        vix_data = collector.collect_vix_data()
        print(f"   ✅ VIX data: {len(vix_data)} records")
        
        # Generate data report
        print(f"\n📋 Generating data report...")
        data_report = IntradayUtils.generate_data_report()
        print(f"   ✅ Data report generated")
        print(f"   📊 Total tables: {data_report.get('overall_summary', {}).get('total_tables', 0)}")
        print(f"   📊 Total records: {data_report.get('overall_summary', {}).get('total_records', 0)}")
        
        collector.close()
        
        print("\n✅ Data collection demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in data collection demo: {e}")
        print(f"❌ Data collection demo failed: {e}")


def feature_engineering_demo():
    """Demonstrate feature engineering functionality."""
    print("\n" + "="*60)
    print("🔧 FEATURE ENGINEERING DEMO")
    print("="*60)
    
    try:
        # Initialize feature engineer
        feature_engineer = IntradayFeatureEngineer()
        
        # Get current timestamp
        current_time = datetime.now()
        
        # Create features for NIFTY
        print(f"\n🔍 Creating features for NIFTY at {current_time}...")
        features = feature_engineer.create_complete_features('NIFTY', current_time)
        
        if features:
            print(f"   ✅ Features created: {len(features)} features")
            
            # Show some key features
            key_features = [
                'atm_ce_delta', 'atm_pe_delta', 'pcr_oi', 'pcr_volume',
                'rsi_14', 'macd', 'bb_deviation', 'vwap_deviation',
                'fii_net', 'dii_net', 'vix_value'
            ]
            
            print(f"\n📊 Key Features:")
            for feature in key_features:
                if feature in features:
                    print(f"   {feature}: {features[feature]:.4f}")
        
        # Get training data for a date range
        print(f"\n📊 Getting training data...")
        start_date = date.today() - timedelta(days=7)
        end_date = date.today()
        
        training_data = feature_engineer.get_training_data('NIFTY', start_date, end_date)
        
        if not training_data.empty:
            print(f"   ✅ Training data: {len(training_data)} samples")
            print(f"   📊 Features: {len(training_data.columns)}")
            print(f"   📊 Labels distribution: {training_data['label'].value_counts().to_dict()}")
        else:
            print(f"   ⚠️ No training data available for the specified date range")
        
        feature_engineer.close()
        
        print("\n✅ Feature engineering demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in feature engineering demo: {e}")
        print(f"❌ Feature engineering demo failed: {e}")


def model_training_demo():
    """Demonstrate model training functionality."""
    print("\n" + "="*60)
    print("🏋️ MODEL TRAINING DEMO")
    print("="*60)
    
    try:
        # Initialize model trainer
        model_trainer = IntradayMLTrainer()
        
        # Train models for NIFTY
        print(f"\n🎯 Training models for NIFTY...")
        start_date = date.today() - timedelta(days=30)  # Last 30 days
        end_date = date.today()
        
        training_results = model_trainer.train_models('NIFTY', start_date, end_date)
        
        if training_results:
            print(f"   ✅ Training completed")
            print(f"   📊 Models trained: {training_results.get('models_trained', 0)}")
            print(f"   🏆 Best model: {training_results.get('best_model', 'N/A')}")
            
            # Show model performances
            performances = training_results.get('model_performances', {})
            if performances:
                print(f"\n📊 Model Performances:")
                for model_name, performance in performances.items():
                    print(f"   {model_name}:")
                    print(f"     Accuracy: {performance.get('accuracy', 0):.3f}")
                    print(f"     F1 Score: {performance.get('f1_score', 0):.3f}")
                    print(f"     Precision: {performance.get('precision', 0):.3f}")
                    print(f"     Recall: {performance.get('recall', 0):.3f}")
            
            # Show feature importance
            feature_importance = training_results.get('feature_importance', {})
            if feature_importance:
                print(f"\n🔍 Top 10 Feature Importance:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:10]):
                    print(f"   {i+1}. {feature}: {importance:.4f}")
        else:
            print(f"   ⚠️ No training results available")
        
        model_trainer.close()
        
        print("\n✅ Model training demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in model training demo: {e}")
        print(f"❌ Model training demo failed: {e}")


def prediction_demo():
    """Demonstrate prediction functionality."""
    print("\n" + "="*60)
    print("🔮 PREDICTION DEMO")
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = IntradayPredictor()
        
        # Load models for NIFTY
        print(f"\n📥 Loading models for NIFTY...")
        if predictor.load_models_for_index('NIFTY'):
            print(f"   ✅ Models loaded successfully")
            
            # Make current prediction
            print(f"\n🔮 Making current prediction...")
            prediction_result = predictor.predict_current('NIFTY')
            
            if prediction_result:
                print(f"   ✅ Prediction completed")
                print(f"   📊 Direction: {prediction_result.get('direction', 'N/A')}")
                print(f"   📊 Confidence: {prediction_result.get('confidence', 0):.3f}")
                print(f"   🤖 Model used: {prediction_result.get('model_used', 'N/A')}")
                print(f"   📈 Market status: {prediction_result.get('market_status', 'N/A')}")
                
                # Show probabilities
                probabilities = prediction_result.get('probabilities', {})
                if probabilities:
                    print(f"   📊 Probabilities:")
                    for direction, prob in probabilities.items():
                        print(f"     {direction}: {prob:.3f}")
            else:
                print(f"   ⚠️ No prediction result available")
            
            # Get feature importance
            print(f"\n🔍 Getting feature importance...")
            feature_importance = predictor.get_feature_importance_summary('NIFTY')
            
            if feature_importance:
                print(f"   ✅ Feature importance retrieved")
                print(f"   🔍 Top 5 features:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:5]):
                    print(f"     {i+1}. {feature}: {importance:.4f}")
            
            # Get model performance
            print(f"\n📊 Getting model performance...")
            model_performance = predictor.get_model_performance('NIFTY')
            
            if model_performance:
                print(f"   ✅ Model performance retrieved")
                print(f"   📊 Available models: {list(model_performance.keys())}")
        else:
            print(f"   ⚠️ Failed to load models")
        
        predictor.close()
        
        print("\n✅ Prediction demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in prediction demo: {e}")
        print(f"❌ Prediction demo failed: {e}")


def system_health_demo():
    """Demonstrate system health check."""
    print("\n" + "="*60)
    print("🏥 SYSTEM HEALTH DEMO")
    print("="*60)
    
    try:
        # Check system health
        print(f"\n🔍 Checking system health...")
        health_status = IntradayUtils.validate_system_health()
        
        if health_status:
            print(f"   ✅ Health check completed")
            print(f"   📊 Overall status: {health_status.get('overall_status', 'N/A')}")
            
            # Show component status
            components = health_status.get('components', {})
            if components:
                print(f"\n🔧 Component Status:")
                for component, status in components.items():
                    status_icon = "✅" if status.get('status') == 'healthy' else "❌"
                    print(f"   {status_icon} {component}: {status.get('status', 'N/A')}")
                    print(f"      Message: {status.get('message', 'N/A')}")
        else:
            print(f"   ⚠️ Health check failed")
        
        print("\n✅ System health demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in system health demo: {e}")
        print(f"❌ System health demo failed: {e}")


def main():
    """Main function to run the intraday ML system."""
    parser = argparse.ArgumentParser(description='Intraday ML Prediction System')
    parser.add_argument('--demo', choices=['data', 'features', 'training', 'prediction', 'health', 'all'],
                       default='all', help='Which demo to run')
    parser.add_argument('--index', choices=['NIFTY', 'BANKNIFTY'], default='NIFTY',
                       help='Index symbol to work with')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🚀 INTRADAY ML PREDICTION SYSTEM")
    print("="*60)
    print(f"📊 Index: {args.index}")
    print(f"🎯 Demo: {args.demo}")
    print("="*60)
    
    try:
        if args.demo == 'data' or args.demo == 'all':
            collect_data_demo()
        
        if args.demo == 'features' or args.demo == 'all':
            feature_engineering_demo()
        
        if args.demo == 'training' or args.demo == 'all':
            model_training_demo()
        
        if args.demo == 'prediction' or args.demo == 'all':
            prediction_demo()
        
        if args.demo == 'health' or args.demo == 'all':
            system_health_demo()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
