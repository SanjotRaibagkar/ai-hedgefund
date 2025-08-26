#!/usr/bin/env python3
"""
Current FNO ML Strategy Analysis
Analyze current predictions for all symbols and categorize by expected returns.
"""

import sys
import os
sys.path.append('./src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import duckdb
from src.ml.fno_ml_strategy import FNOMLStrategy
from src.tools.enhanced_api import get_fno_bhav_copy_data

def get_all_fno_symbols():
    """Get all unique FNO symbols from the database."""
    try:
        db_path = "data/comprehensive_equity.duckdb"
        conn = duckdb.connect(db_path)
        
        # Get all unique ticker symbols from fno_bhav_copy
        query = """
        SELECT DISTINCT TckrSymb 
        FROM fno_bhav_copy 
        WHERE TckrSymb IS NOT NULL 
        AND TckrSymb != ''
        ORDER BY TckrSymb
        """
        
        result = conn.execute(query).fetchdf()
        conn.close()
        
        symbols = result['TckrSymb'].tolist()
        logger.info(f"Found {len(symbols)} FNO symbols")
        return symbols
        
    except Exception as e:
        logger.error(f"Error getting FNO symbols: {e}")
        return []

def analyze_symbol(strategy, symbol):
    """Analyze a single symbol and return prediction data."""
    try:
        logger.info(f"Analyzing {symbol}")
        
        # Set date range for current analysis (last 30 days for training, next 1 day for prediction)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # First, train models for the symbol
        strategy.train_models(symbol, start_date, end_date)
        
        # Get predictions for the symbol
        predictions = strategy.predict_returns(symbol, start_date, end_date)
        
        if not predictions or 'error' in predictions:
            logger.warning(f"No predictions for {symbol}")
            return None
        
        # Get the latest prediction for 1-day horizon
        if '1d' in predictions and predictions['1d']:
            pred_data = predictions['1d']
            if 'predictions' in pred_data and not pred_data['predictions'].empty:
                latest_pred = pred_data['predictions'].iloc[-1]
                
                return {
                    'Symbol': symbol,
                    'Expected_Return_1d': latest_pred.get('predicted_return', 0),
                    'Confidence': latest_pred.get('confidence', 0),
                    'Signal': 'BUY' if latest_pred.get('predicted_return', 0) > 0 else 'SELL',
                    'Signal_Strength': 'STRONG' if abs(latest_pred.get('predicted_return', 0)) > 0.03 else 'MODERATE',
                    'Analysis_Date': datetime.now().strftime('%Y-%m-%d')
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

def categorize_by_returns(df):
    """Categorize symbols into quartiles based on expected returns."""
    try:
        # Remove any rows with NaN expected returns
        df_clean = df.dropna(subset=['Expected_Return_1d'])
        
        if len(df_clean) == 0:
            return df
        
        # Calculate quartiles
        q25 = df_clean['Expected_Return_1d'].quantile(0.25)
        q50 = df_clean['Expected_Return_1d'].quantile(0.50)
        q75 = df_clean['Expected_Return_1d'].quantile(0.75)
        
        # Create quartile categories
        def get_quartile_category(row):
            ret = row['Expected_Return_1d']
            if ret >= q75:
                return 'Q4 (Highest)'
            elif ret >= q50:
                return 'Q3 (High)'
            elif ret >= q25:
                return 'Q2 (Medium)'
            else:
                return 'Q1 (Lowest)'
        
        # Create return-based categories
        def get_return_category(row):
            ret = row['Expected_Return_1d']
            if ret >= 0.05:  # 5% or more
                return 'Strong Buy (5%+)'
            elif ret >= 0.03:  # 3-5%
                return 'Buy (3-5%)'
            elif ret >= 0.01:  # 1-3%
                return 'Moderate Buy (1-3%)'
            elif ret >= -0.01:  # -1% to 1%
                return 'Neutral (-1% to 1%)'
            elif ret >= -0.03:  # -3% to -1%
                return 'Moderate Sell (-3% to -1%)'
            elif ret >= -0.05:  # -5% to -3%
                return 'Sell (-5% to -3%)'
            else:  # Less than -5%
                return 'Strong Sell (<-5%)'
        
        df_clean['Quartile'] = df_clean.apply(get_quartile_category, axis=1)
        df_clean['Return_Category'] = df_clean.apply(get_return_category, axis=1)
        
        # Add quartile statistics
        df_clean['Q25'] = q25
        df_clean['Q50'] = q50
        df_clean['Q75'] = q75
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error categorizing returns: {e}")
        return df

def create_excel_report(df, filename):
    """Create Excel report with multiple sheets."""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='All_Predictions', index=False)
            
            # Summary by quartile
            if 'Quartile' in df.columns:
                quartile_summary = df.groupby('Quartile').agg({
                    'Symbol': 'count',
                    'Expected_Return_1d': ['mean', 'min', 'max'],
                    'Confidence': 'mean'
                }).round(4)
                quartile_summary.columns = ['Count', 'Avg_Return', 'Min_Return', 'Max_Return', 'Avg_Confidence']
                quartile_summary.to_excel(writer, sheet_name='Quartile_Summary')
            
            # Summary by return category
            if 'Return_Category' in df.columns:
                category_summary = df.groupby('Return_Category').agg({
                    'Symbol': 'count',
                    'Expected_Return_1d': ['mean', 'min', 'max'],
                    'Confidence': 'mean'
                }).round(4)
                category_summary.columns = ['Count', 'Avg_Return', 'Min_Return', 'Max_Return', 'Avg_Confidence']
                category_summary.to_excel(writer, sheet_name='Return_Category_Summary')
            
            # Signal summary
            if 'Signal' in df.columns:
                signal_summary = df.groupby('Signal').agg({
                    'Symbol': 'count',
                    'Expected_Return_1d': 'mean',
                    'Confidence': 'mean'
                }).round(4)
                signal_summary.columns = ['Count', 'Avg_Return', 'Avg_Confidence']
                signal_summary.to_excel(writer, sheet_name='Signal_Summary')
            
            # Top performers (highest expected returns)
            top_performers = df.nlargest(20, 'Expected_Return_1d')[['Symbol', 'Expected_Return_1d', 'Confidence', 'Signal', 'Return_Category']]
            top_performers.to_excel(writer, sheet_name='Top_20_Performers', index=False)
            
            # Bottom performers (lowest expected returns)
            bottom_performers = df.nsmallest(20, 'Expected_Return_1d')[['Symbol', 'Expected_Return_1d', 'Confidence', 'Signal', 'Return_Category']]
            bottom_performers.to_excel(writer, sheet_name='Bottom_20_Performers', index=False)
            
            # High confidence predictions
            high_confidence = df[df['Confidence'] >= 0.75].sort_values('Confidence', ascending=False)
            if len(high_confidence) > 0:
                high_confidence.to_excel(writer, sheet_name='High_Confidence_Predictions', index=False)
        
        logger.info(f"Excel report saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Error creating Excel report: {e}")

def main():
    """Main function to run current analysis."""
    print("🚀 FNO ML Strategy Current Analysis")
    print("=" * 80)
    
    try:
        # Initialize strategy with enhanced configuration
        strategy_config = {
            'min_data_points': 15,  # Reduced for current analysis
            'min_oi_threshold': 50,  # Reduced for current analysis
            'min_volume_threshold': 50,  # Reduced for current analysis
            'confidence_threshold': 0.75,
            'price_change_threshold': 0.02,
            'ensemble_voting': True,
            'min_liquidity_score': 0.1  # Reduced for current analysis
        }
        
        ml_config = {
            'model_type': 'ensemble',
            'test_size': 0.2,
            'random_state': 42,
            'prediction_horizons': [1, 5, 21],
            'feature_selection': True,
            'ensemble_models': ['xgboost', 'lightgbm', 'random_forest', 'linear'],
            'ensemble_weights': [0.4, 0.3, 0.2, 0.1]
        }
        
        strategy = FNOMLStrategy(strategy_config=strategy_config, ml_config=ml_config)
        
        # Get all FNO symbols
        symbols = get_all_fno_symbols()
        
        if not symbols:
            print("❌ No FNO symbols found in database")
            return
        
        print(f"📊 Analyzing {len(symbols)} FNO symbols...")
        
        # Analyze each symbol
        results = []
        successful = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"Progress: {i}/{len(symbols)} - {symbol}")
            
            result = analyze_symbol(strategy, symbol)
            if result:
                results.append(result)
                successful += 1
        
        if not results:
            print("❌ No successful predictions generated")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Categorize by returns
        df_categorized = categorize_by_returns(df)
        
        # Create Excel report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/fno_current_analysis_{timestamp}.xlsx"
        
        # Ensure directory exists
        os.makedirs('data', exist_ok=True)
        
        create_excel_report(df_categorized, filename)
        
        # Print summary
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {filename}")
        print(f"📊 Symbols analyzed: {successful}/{len(symbols)} ({successful/len(symbols)*100:.1f}%)")
        
        if len(df_categorized) > 0:
            print(f"\n📈 Return Statistics:")
            print(f"   - Average Expected Return: {df_categorized['Expected_Return_1d'].mean():.4f}")
            print(f"   - Min Expected Return: {df_categorized['Expected_Return_1d'].min():.4f}")
            print(f"   - Max Expected Return: {df_categorized['Expected_Return_1d'].max():.4f}")
            print(f"   - Average Confidence: {df_categorized['Confidence'].mean():.4f}")
            
            if 'Return_Category' in df_categorized.columns:
                print(f"\n📊 Return Categories:")
                category_counts = df_categorized['Return_Category'].value_counts()
                for category, count in category_counts.items():
                    print(f"   - {category}: {count} symbols")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
