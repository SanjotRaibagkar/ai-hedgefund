#!/usr/bin/env python3
"""
Batch Nifty Options Predictor
Analyze multiple dates and times for pattern recognition
"""

import pandas as pd
from options_analyzer_v2 import OptionsAnalyzerV2
from pathlib import Path
import json
from datetime import datetime, timedelta

def batch_predict_dates(dates, times=None, parquet_path="../../../data/options_parquet", output_file=None):
    """
    Predict next moves for multiple dates and times
    
    Args:
        dates (list): List of dates in "YYYYMMDD" format
        times (list): List of times to analyze (default: ["09:30:00", "11:00:00", "13:30:00", "15:00:00"])
        parquet_path (str): Path to parquet files folder
        output_file (str): Optional CSV file to save results
    
    Returns:
        pd.DataFrame: Results dataframe
    """
    if times is None:
        times = ["09:30:00", "11:00:00", "13:30:00", "15:00:00"]
    
    print(f"🔮 BATCH PREDICTING FOR {len(dates)} DATES")
    print(f"⏰ Analyzing {len(times)} time slots per date")
    print("=" * 80)
    
    results = []
    
    for date in dates:
        print(f"\n📅 Analyzing {date}...")
        
        try:
            # Initialize analyzer
            analyzer = OptionsAnalyzerV2(parquet_path)
            
            # Load data for the date
            analyzer.load_data_for_date(date)
            
            for time in times:
                try:
                    # Generate prediction
                    result = analyzer.generate_prediction_signal(time)
                    
                    if 'error' not in result:
                        # Extract key metrics
                        row = {
                            'date': date,
                            'time': time,
                            'direction': result['direction'],
                            'confidence': result['confidence'],
                            'signal_score': result['signal_score'],
                            'spot_estimate': result['spot_estimate'],
                            'pcr_oi': result['detailed_metrics']['pcr'].get('pcr_oi', 0),
                            'pcr_sentiment': result['detailed_metrics']['pcr'].get('sentiment', 'N/A'),
                            'max_pain': result['detailed_metrics']['max_pain'],
                            'flow_bias': result['detailed_metrics']['flows'].get('flow_bias', 'N/A'),
                            'gamma_environment': result['detailed_metrics']['gamma'].get('gamma_interpretation', 'N/A'),
                            'iv_skew': result['detailed_metrics']['iv_metrics'].get('iv_skew', 0),
                            'key_factors': ', '.join(result['components'])
                        }
                        
                        results.append(row)
                        print(f"   ✅ {time}: {result['direction']} ({result['confidence']}) - Score: {result['signal_score']:.3f}")
                    else:
                        print(f"   ❌ {time}: {result['error']}")
                        
                except Exception as e:
                    print(f"   ❌ {time}: Error - {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error loading data for {date}: {e}")
            continue
    
    # Create results dataframe
    if results:
        df = pd.DataFrame(results)
        
        print(f"\n📊 BATCH ANALYSIS COMPLETE!")
        print(f"✅ Successfully analyzed {len(df)} data points")
        
        # Summary statistics
        print(f"\n📈 DIRECTION BREAKDOWN:")
        direction_counts = df['direction'].value_counts()
        for direction, count in direction_counts.items():
            print(f"   {direction}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\n🔒 CONFIDENCE BREAKDOWN:")
        confidence_counts = df['confidence'].value_counts()
        for confidence, count in confidence_counts.items():
            print(f"   {confidence}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\n📊 SIGNAL SCORE STATISTICS:")
        print(f"   Average Score: {df['signal_score'].mean():.3f}")
        print(f"   Min Score: {df['signal_score'].min():.3f}")
        print(f"   Max Score: {df['signal_score'].max():.3f}")
        
        # Save to CSV if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        return df
    else:
        print("❌ No results generated")
        return None

def predict_week_range(start_date, end_date, times=None, parquet_path="../../../data/options_parquet"):
    """
    Predict for a range of dates
    
    Args:
        start_date (str): Start date in "YYYYMMDD" format
        end_date (str): End date in "YYYYMMDD" format
        times (list): List of times to analyze
        parquet_path (str): Path to parquet files folder
    
    Returns:
        pd.DataFrame: Results dataframe
    """
    # Generate date range
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    return batch_predict_dates(dates, times, parquet_path)

def main():
    """Main function with examples"""
    print("🔮 BATCH NIFTY OPTIONS PREDICTOR")
    print("=" * 80)
    
    # Example 1: Analyze specific dates
    print("\n📊 EXAMPLE 1: Analyze specific dates")
    specific_dates = ["20250829", "20250828", "20250827"]
    result1 = batch_predict_dates(specific_dates, output_file="specific_dates_predictions.csv")
    
    # Example 2: Analyze date range
    print(f"\n{'='*80}")
    print("\n📊 EXAMPLE 2: Analyze date range")
    result2 = predict_week_range("20250825", "20250829", output_file="week_range_predictions.csv")
    
    # Example 3: Custom time slots
    print(f"\n{'='*80}")
    print("\n📊 EXAMPLE 3: Custom time slots")
    custom_times = ["10:00:00", "12:00:00", "14:00:00"]
    result3 = batch_predict_dates(["20250829"], custom_times, output_file="custom_times_predictions.csv")

if __name__ == "__main__":
    main()
