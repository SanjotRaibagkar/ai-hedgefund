#!/usr/bin/env python3
"""
Nifty Options Next Move Predictor
A simple tool to predict the next move using options data
"""

import pandas as pd
from options_analyzer_v2 import OptionsAnalyzerV2
from pathlib import Path
import argparse
from datetime import datetime

def predict_next_move(date_str, time_str="13:30:00", spot_price=None, parquet_path="../../../data/options_parquet"):
    """
    Predict the next move for a specific date and time
    
    Args:
        date_str (str): Date in format "YYYYMMDD" (e.g., "20250829")
        time_str (str): Time to analyze (default: "13:30:00")
        spot_price (float): Optional spot price override
        parquet_path (str): Path to parquet files folder
    
    Returns:
        dict: Prediction results
    """
    print(f"🔮 PREDICTING NEXT MOVE FOR {date_str} at {time_str}")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = OptionsAnalyzerV2(parquet_path)
        
        # Load data for the specified date
        analyzer.load_data_for_date(date_str)
        print(f"✅ Data loaded successfully for {date_str}")
        
        # Generate prediction signal
        if spot_price:
            print(f"📍 Using provided spot price: {spot_price:,}")
            result = analyzer.generate_prediction_signal(time_str, spot_price)
        else:
            print(f"📍 Estimating spot price from options data...")
            result = analyzer.generate_prediction_signal(time_str)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return None
        
        # Display prediction results
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Direction: {result['direction']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Signal Score: {result['signal_score']:.3f}")
        print(f"   Key Factors: {', '.join(result['components'])}")
        
        # Display detailed metrics
        print(f"\n📊 DETAILED ANALYSIS:")
        
        # PCR Analysis
        pcr_data = result['detailed_metrics']['pcr']
        if 'error' not in pcr_data:
            print(f"   📈 PCR (OI): {pcr_data.get('pcr_oi', 0):.3f} - {pcr_data.get('sentiment', 'N/A')}")
            print(f"   📊 PCR (Volume): {pcr_data.get('pcr_volume', 0):.3f}")
        
        # Max Pain Analysis
        max_pain = result['detailed_metrics']['max_pain']
        if max_pain is not None:
            pain_distance = (result['spot_estimate'] - max_pain) / result['spot_estimate'] * 100
            print(f"   🎯 Max Pain: {max_pain:,} (Distance: {pain_distance:.2f}%)")
        
        # Flow Analysis
        flow_data = result['detailed_metrics']['flows']
        if 'error' not in flow_data:
            print(f"   🔄 Flow Bias: {flow_data.get('flow_bias', 'N/A')}")
            print(f"   📊 Flow Intensity: {flow_data.get('flow_intensity', 0):,.0f}")
        
        # Gamma Analysis
        gamma_data = result['detailed_metrics']['gamma']
        if 'error' not in gamma_data:
            print(f"   ⚡ Gamma Environment: {gamma_data.get('gamma_interpretation', 'N/A')}")
            print(f"   📊 Net Gamma Exposure: {gamma_data.get('net_gamma_exposure', 0):,.2f}")
        
        # IV Analysis
        iv_data = result['detailed_metrics']['iv_metrics']
        if 'error' not in iv_data:
            print(f"   📈 IV Skew: {iv_data.get('iv_skew', 0):.2f} - {iv_data.get('skew_interpretation', 'N/A')}")
        
        # Support/Resistance Levels
        levels = result['detailed_metrics']['support_resistance']
        if 'error' not in levels:
            print(f"\n🏗️ KEY LEVELS:")
            print(f"   🔴 Resistance: {levels['resistance']['Strike_Price'].iloc[0]:,} (Call OI: {levels['resistance']['CALLS_OI'].iloc[0]:,.0f})")
            print(f"   🟢 Support: {levels['support']['Strike_Price'].iloc[0]:,} (Put OI: {levels['support']['PUTS_OI'].iloc[0]:,.0f})")
        
        # Momentum Analysis
        momentum = analyzer.get_intraday_momentum(time_str)
        if 'error' not in momentum:
            print(f"\n📈 MOMENTUM:")
            print(f"   Trend: {momentum.get('momentum', 'N/A')}")
            print(f"   PCR Trend: {momentum.get('pcr_trend', 0):.3f}")
        
        # Trading Recommendations
        print(f"\n💡 TRADING RECOMMENDATIONS:")
        if result['direction'] == "BULLISH":
            if result['confidence'] == "HIGH":
                print(f"   🚀 Strong bullish signal - Consider buying calls or selling puts")
            else:
                print(f"   📈 Moderate bullish signal - Cautious call buying")
        elif result['direction'] == "BEARISH":
            if result['confidence'] == "HIGH":
                print(f"   📉 Strong bearish signal - Consider buying puts or selling calls")
            else:
                print(f"   🔻 Moderate bearish signal - Cautious put buying")
        else:
            print(f"   ⚖️ Neutral signal - Range-bound trading or wait for clearer signals")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Predict next move using Nifty options data')
    parser.add_argument('date', help='Date in YYYYMMDD format (e.g., 20250829)')
    parser.add_argument('--time', default='13:30:00', help='Time to analyze (default: 13:30:00)')
    parser.add_argument('--spot', type=float, help='Override spot price')
    parser.add_argument('--path', default='../../../data/options_parquet', help='Path to parquet files')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y%m%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.date}. Use YYYYMMDD format (e.g., 20250829)")
        return
    
    # Run prediction
    result = predict_next_move(args.date, args.time, args.spot, args.path)
    
    if result:
        print(f"\n✅ Prediction completed successfully!")
        print(f"📅 Date: {args.date}")
        print(f"⏰ Time: {args.time}")
        print(f"🎯 Next Move: {result['direction']} ({result['confidence']} confidence)")

if __name__ == "__main__":
    # Run command line interface
    main()
