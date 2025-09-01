#!/usr/bin/env python3
"""
Analyze prediction bias in backtest results
"""

import pandas as pd

def analyze_predictions():
    """Analyze prediction distribution and bias"""
    
    # Read the CSV file
    df = pd.read_csv('fno_comprehensive_backtest_results_FIXED_20250831_173710.csv')
    
    print("=== PREDICTION BIAS ANALYSIS ===")
    print(f"Total records: {len(df)}")
    
    print("\n1. PREDICTED DIRECTION DISTRIBUTION:")
    print(df['predicted_direction'].value_counts())
    
    print("\n2. ACTUAL DIRECTION DISTRIBUTION:")
    print(df['actual_direction'].value_counts())
    
    print("\n3. ML PREDICTION STATISTICS:")
    print(f"Min: {df['ml_prediction'].min():.3f}")
    print(f"Max: {df['ml_prediction'].max():.3f}")
    print(f"Mean: {df['ml_prediction'].mean():.3f}")
    print(f"Median: {df['ml_prediction'].median():.3f}")
    print(f"Std: {df['ml_prediction'].std():.3f}")
    
    print("\n4. PREDICTION RANGE BREAKDOWN:")
    print(f"Predictions < 0.4: {len(df[df['ml_prediction'] < 0.4])} ({(len(df[df['ml_prediction'] < 0.4])/len(df)*100):.1f}%)")
    print(f"Predictions 0.4-0.5: {len(df[(df['ml_prediction'] >= 0.4) & (df['ml_prediction'] < 0.5)])} ({(len(df[(df['ml_prediction'] >= 0.4) & (df['ml_prediction'] < 0.5)])/len(df)*100):.1f}%)")
    print(f"Predictions 0.5-0.6: {len(df[(df['ml_prediction'] >= 0.5) & (df['ml_prediction'] < 0.6)])} ({(len(df[(df['ml_prediction'] >= 0.5) & (df['ml_prediction'] < 0.6)])/len(df)*100):.1f}%)")
    print(f"Predictions 0.6-0.7: {len(df[(df['ml_prediction'] >= 0.6) & (df['ml_prediction'] < 0.7)])} ({(len(df[(df['ml_prediction'] >= 0.6) & (df['ml_prediction'] < 0.7)])/len(df)*100):.1f}%)")
    print(f"Predictions 0.7-0.8: {len(df[(df['ml_prediction'] >= 0.7) & (df['ml_prediction'] < 0.8)])} ({(len(df[(df['ml_prediction'] >= 0.7) & (df['ml_prediction'] < 0.8)])/len(df)*100):.1f}%)")
    print(f"Predictions > 0.8: {len(df[df['ml_prediction'] >= 0.8])} ({(len(df[df['ml_prediction'] >= 0.8])/len(df)*100):.1f}%)")
    
    print("\n5. SAMPLE OF DOWN PREDICTIONS (if any):")
    down_predictions = df[df['predicted_direction'] == 'DOWN']
    if len(down_predictions) > 0:
        print(down_predictions[['symbol', 'ml_prediction', 'predicted_direction', 'actual_direction', 'actual_return']].head(10))
    else:
        print("No DOWN predictions found!")
    
    print("\n6. SAMPLE OF ACTUAL DOWN MOVEMENTS:")
    actual_down = df[df['actual_direction'] == 'DOWN']
    print(f"Total actual DOWN movements: {len(actual_down)}")
    if len(actual_down) > 0:
        print(actual_down[['symbol', 'ml_prediction', 'predicted_direction', 'actual_direction', 'actual_return']].head(10))

if __name__ == "__main__":
    analyze_predictions()

