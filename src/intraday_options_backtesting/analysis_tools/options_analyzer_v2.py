#!/usr/bin/env python3
"""
Options Analyzer V2 for Intraday Options Backtesting
Implements comprehensive options analysis logic for prediction signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class OptionsAnalyzerV2:
    def __init__(self, parquet_folder_path):
        """
        Initialize with parquet folder path
        parquet_folder_path: Path to the parquet files folder
        """
        self.parquet_folder_path = Path(parquet_folder_path)
        self.df = None
        self.current_date = None
        
    def load_data_for_date(self, date_str):
        """
        Load all parquet data for a specific date
        """
        date_folder = self.parquet_folder_path / date_str
        if not date_folder.exists():
            raise FileNotFoundError(f"No data found for date {date_str}")
        
        # Load all NIFTY files for the date
        nifty_files = list(date_folder.glob("NIFTY_*.parquet"))
        if not nifty_files:
            raise FileNotFoundError(f"No NIFTY files found for date {date_str}")
        
        # Read and combine all files
        all_data = []
        for file_path in sorted(nifty_files):
            try:
                df = pd.read_parquet(file_path)
                df['source_file'] = file_path.name
                df['file_timestamp'] = file_path.stem.split('_')[-1]
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"No valid data could be loaded for date {date_str}")
        
        # Combine all data
        self.df = pd.concat(all_data, ignore_index=True)
        self.current_date = date_str
        
        # Convert columns - note: date column is already datetime.date
        self.df['Fetch_Time'] = pd.to_datetime(self.df['Fetch_Time'])
        self.df['Expiry_Date'] = pd.to_datetime(self.df['Expiry_Date'])
        # Keep date column as is since it's already datetime.date
        
        print(f"✅ Loaded {len(self.df)} records from {len(nifty_files)} files for {date_str}")
        
    def get_snapshot_at_time(self, target_time="13:30:00", target_date=None):
        """
        Get the latest options chain snapshot at or before target_time
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data_for_date() first.")
        
        if target_date is None:
            target_date = self.current_date
        
        # Convert target_date string to datetime.date object for comparison
        if isinstance(target_date, str):
            target_date_obj = pd.to_datetime(target_date).date()
        else:
            target_date_obj = target_date
        
        target_datetime = pd.to_datetime(f"{target_date} {target_time}")
        
        # Get the latest snapshot before or at target time
        # Compare date objects directly
        snapshot = self.df[
            (self.df['date'] == target_date_obj) & 
            (self.df['Fetch_Time'] <= target_datetime)
        ].groupby(['Symbol', 'Expiry_Date', 'Strike_Price']).last().reset_index()
        
        return snapshot
    
    def calculate_pcr_indicators(self, snapshot):
        """
        Calculate Put-Call Ratio indicators
        """
        if len(snapshot) == 0:
            return {'error': 'Empty snapshot'}
        
        # Current expiry (nearest)
        current_expiry = snapshot['Expiry_Date'].min()
        current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
        
        # PCR by Open Interest
        total_call_oi = current_exp_data['CALLS_OI'].sum()
        total_put_oi = current_exp_data['PUTS_OI'].sum()
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # PCR by Volume
        total_call_vol = current_exp_data['CALLS_Volume'].sum()
        total_put_vol = current_exp_data['PUTS_Volume'].sum()
        pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # PCR by Change in OI
        total_call_chng_oi = current_exp_data['CALLS_Chng_in_OI'].sum()
        total_put_chng_oi = current_exp_data['PUTS_Chng_in_OI'].sum()
        pcr_chng_oi = total_put_chng_oi / total_call_chng_oi if total_call_chng_oi > 0 else 0
        
        return {
            'pcr_oi': pcr_oi,
            'pcr_volume': pcr_volume,
            'pcr_chng_oi': pcr_chng_oi,
            'sentiment': self._interpret_pcr(pcr_oi, pcr_volume)
        }
    
    def find_max_pain(self, snapshot):
        """
        Calculate Max Pain level
        """
        if len(snapshot) == 0:
            return None
        
        current_expiry = snapshot['Expiry_Date'].min()
        current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
        
        strikes = current_exp_data['Strike_Price'].unique()
        max_pain_values = []
        
        for strike in strikes:
            total_pain = 0
            
            for _, row in current_exp_data.iterrows():
                strike_price = row['Strike_Price']
                call_oi = row['CALLS_OI']
                put_oi = row['PUTS_OI']
                
                # Calculate pain for calls (ITM when spot > strike)
                if strike > strike_price:
                    total_pain += call_oi * (strike - strike_price)
                
                # Calculate pain for puts (ITM when spot < strike)
                if strike < strike_price:
                    total_pain += put_oi * (strike_price - strike)
            
            max_pain_values.append({'strike': strike, 'total_pain': total_pain})
        
        max_pain_df = pd.DataFrame(max_pain_values)
        if len(max_pain_df) > 0:
            max_pain_strike = max_pain_df.loc[max_pain_df['total_pain'].idxmin(), 'strike']
            return max_pain_strike
        return None
    
    def identify_support_resistance(self, snapshot):
        """
        Identify key support and resistance levels from OI
        """
        if len(snapshot) == 0:
            return {'error': 'Empty snapshot'}
        
        current_expiry = snapshot['Expiry_Date'].min()
        current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
        
        # Sort by total OI
        current_exp_data['Total_OI'] = current_exp_data['CALLS_OI'] + current_exp_data['PUTS_OI']
        oi_sorted = current_exp_data.sort_values('Total_OI', ascending=False)
        
        # Top 5 strikes by total OI
        key_levels = oi_sorted.head(5)[['Strike_Price', 'CALLS_OI', 'PUTS_OI', 'Total_OI']]
        
        # Resistance levels (high Call OI)
        resistance_levels = current_exp_data.nlargest(3, 'CALLS_OI')[['Strike_Price', 'CALLS_OI']]
        
        # Support levels (high Put OI)
        support_levels = current_exp_data.nlargest(3, 'PUTS_OI')[['Strike_Price', 'PUTS_OI']]
        
        return {
            'key_levels': key_levels,
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def calculate_gamma_exposure(self, snapshot, spot_price):
        """
        Calculate Dealer Gamma Exposure (simplified)
        """
        if len(snapshot) == 0:
            return {'error': 'Empty snapshot'}
        
        current_expiry = snapshot['Expiry_Date'].min()
        current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
        
        # Simplified gamma calculation (assumes 0.01 gamma per 100 points OTM)
        gamma_exposure = []
        
        for _, row in current_exp_data.iterrows():
            try:
                strike = float(row['Strike_Price'])
                call_oi = float(row['CALLS_OI']) if pd.notna(row['CALLS_OI']) else 0.0
                put_oi = float(row['PUTS_OI']) if pd.notna(row['PUTS_OI']) else 0.0
                
                # Skip if invalid data
                if pd.isna(strike) or pd.isna(call_oi) or pd.isna(put_oi):
                    continue
                
                # Simplified gamma approximation
                moneyness = abs(strike - spot_price) / spot_price
                gamma_approx = max(0.01 * (1 - moneyness * 10), 0.001)  # Simplified
                
                # Net gamma exposure (dealer perspective)
                call_gamma_exp = -call_oi * gamma_approx  # Dealers short gamma on calls
                put_gamma_exp = -put_oi * gamma_approx    # Dealers short gamma on puts
                
                gamma_exposure.append({
                    'strike': strike,
                    'call_gamma_exp': call_gamma_exp,
                    'put_gamma_exp': put_gamma_exp,
                    'net_gamma_exp': call_gamma_exp + put_gamma_exp
                })
            except (ValueError, TypeError) as e:
                # Skip problematic rows
                continue
        
        if not gamma_exposure:
            return {'error': 'No valid gamma data calculated'}
        
        gamma_df = pd.DataFrame(gamma_exposure)
        net_gamma = gamma_df['net_gamma_exp'].sum()
        
        return {
            'net_gamma_exposure': net_gamma,
            'gamma_by_strike': gamma_df,
            'gamma_interpretation': 'Positive Gamma (Stabilizing)' if net_gamma > 0 else 'Negative Gamma (Amplifying)'
        }
    
    def analyze_flow_dynamics(self, snapshot, lookback_minutes=30):
        """
        Analyze recent flow dynamics
        """
        if len(snapshot) == 0:
            return {'error': 'Empty snapshot'}
        
        current_time = snapshot['Fetch_Time'].max()
        lookback_time = current_time - timedelta(minutes=lookback_minutes)
        
        # Get recent data for flow analysis
        recent_data = self.df[
            (self.df['Fetch_Time'] >= lookback_time) & 
            (self.df['Fetch_Time'] <= current_time)
        ]
        
        if len(recent_data) < 2:
            return {'status': 'Insufficient data for flow analysis'}
        
        # Aggregate flow metrics
        current_expiry = snapshot['Expiry_Date'].min()
        recent_exp_data = recent_data[recent_data['Expiry_Date'] == current_expiry]
        
        # Calculate net flows
        call_oi_flow = recent_exp_data.groupby('Strike_Price')['CALLS_Chng_in_OI'].sum()
        put_oi_flow = recent_exp_data.groupby('Strike_Price')['PUTS_Chng_in_OI'].sum()
        
        call_vol_flow = recent_exp_data.groupby('Strike_Price')['CALLS_Volume'].sum()
        put_vol_flow = recent_exp_data.groupby('Strike_Price')['PUTS_Volume'].sum()
        
        # Net buying pressure
        net_call_pressure = call_oi_flow.sum() + call_vol_flow.sum()
        net_put_pressure = put_oi_flow.sum() + put_vol_flow.sum()
        
        # Calculate flow bias with threshold
        pressure_diff = net_call_pressure - net_put_pressure
        pressure_ratio = abs(pressure_diff) / max(net_call_pressure, net_put_pressure)
        
        if pressure_ratio > 0.1:  # More than 10% difference
            flow_bias = 'Bullish' if pressure_diff > 0 else 'Bearish'
        else:
            flow_bias = 'Neutral'
        
        return {
            'net_call_pressure': net_call_pressure,
            'net_put_pressure': net_put_pressure,
            'flow_bias': flow_bias,
            'flow_intensity': abs(pressure_diff),
            'pressure_ratio': pressure_ratio
        }
    
    def calculate_iv_signals(self, snapshot):
        """
        Calculate IV-based signals
        """
        if len(snapshot) == 0:
            return {'error': 'Empty snapshot'}
        
        current_expiry = snapshot['Expiry_Date'].min()
        current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
        
        # ATM IV (simplified - using median strike)
        median_strike = current_exp_data['Strike_Price'].median()
        atm_data = current_exp_data.iloc[(current_exp_data['Strike_Price'] - median_strike).abs().argsort()[:1]]
        
        atm_iv_call = atm_data['CALLS_IV'].iloc[0] if len(atm_data) > 0 else 0
        atm_iv_put = atm_data['PUTS_IV'].iloc[0] if len(atm_data) > 0 else 0
        
        # IV Skew (25 delta puts vs calls)
        otm_puts = current_exp_data[current_exp_data['Strike_Price'] < median_strike]
        otm_calls = current_exp_data[current_exp_data['Strike_Price'] > median_strike]
        
        avg_put_iv = otm_puts['PUTS_IV'].mean() if len(otm_puts) > 0 else 0
        avg_call_iv = otm_calls['CALLS_IV'].mean() if len(otm_calls) > 0 else 0
        
        iv_skew = avg_put_iv - avg_call_iv
        
        return {
            'atm_iv_call': atm_iv_call,
            'atm_iv_put': atm_iv_put,
            'iv_skew': iv_skew,
            'skew_interpretation': 'Put skew (Bearish)' if iv_skew > 2 else 'Call skew (Bullish)' if iv_skew < -2 else 'Neutral'
        }
    
    def generate_prediction_signal(self, target_time="13:30:00", spot_price=None):
        """
        Main function to generate prediction signal
        """
        # Get snapshot
        snapshot = self.get_snapshot_at_time(target_time)
        
        if len(snapshot) == 0:
            return {'error': 'No data available for specified time'}
        
        # Estimate spot price if not provided
        if spot_price is None:
            # Use the strike with highest total OI as proxy for spot (more accurate)
            current_expiry = snapshot['Expiry_Date'].min()
            current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
            current_exp_data['Total_OI'] = current_exp_data['CALLS_OI'] + current_exp_data['PUTS_OI']
            
            # Find the strike with highest total OI (usually closest to actual spot)
            max_oi_strike = current_exp_data.loc[current_exp_data['Total_OI'].idxmax(), 'Strike_Price']
            spot_price = float(max_oi_strike)
            
            print(f"   📍 Estimated spot price: {spot_price:,} (from max OI strike)")
        
        # Calculate all indicators
        pcr_data = self.calculate_pcr_indicators(snapshot)
        max_pain = self.find_max_pain(snapshot)
        levels = self.identify_support_resistance(snapshot)
        gamma_data = self.calculate_gamma_exposure(snapshot, spot_price)
        flow_data = self.analyze_flow_dynamics(snapshot)
        iv_data = self.calculate_iv_signals(snapshot)
        
        # Generate composite signal
        signal_score = 0
        signal_components = []
        
        # PCR signals (weight: 25%)
        if pcr_data.get('pcr_oi', 1) < 0.8:
            signal_score += 0.25
            signal_components.append("Bullish PCR")
        elif pcr_data.get('pcr_oi', 1) > 1.2:
            signal_score -= 0.25
            signal_components.append("Bearish PCR")
        
        # Max Pain signal (weight: 20%)
        if max_pain is not None:
            pain_distance = (spot_price - max_pain) / spot_price * 100
            if pain_distance > 1:  # Spot significantly above max pain
                signal_score -= 0.15
                signal_components.append("Above Max Pain (Bearish)")
            elif pain_distance < -1:  # Spot significantly below max pain
                signal_score += 0.15
                signal_components.append("Below Max Pain (Bullish)")
            else:  # Spot near max pain (neutral)
                signal_components.append("Near Max Pain (Neutral)")
        
        # Flow analysis (weight: 30%)
        if 'flow_bias' in flow_data:
            print(f"   🔄 Flow Analysis: {flow_data['flow_bias']} (Call: {flow_data.get('net_call_pressure', 0):,.0f}, Put: {flow_data.get('net_put_pressure', 0):,.0f}, Ratio: {flow_data.get('pressure_ratio', 0):.1%})")
            if flow_data['flow_bias'] == 'Bullish':
                signal_score += 0.3
                signal_components.append("Bullish Flow")
            else:
                signal_score -= 0.3
                signal_components.append("Bearish Flow")
        
        # IV Skew (weight: 15%)
        if iv_data.get('iv_skew', 0) > 2:
            signal_score -= 0.15
            signal_components.append("Put Skew (Bearish)")
        elif iv_data.get('iv_skew', 0) < -2:
            signal_score += 0.15
            signal_components.append("Call Skew (Bullish)")
        
        # Gamma environment (weight: 10%)
        if 'error' not in gamma_data and gamma_data.get('net_gamma_exposure', 0) > 0:
            signal_components.append("Positive Gamma (Range-bound)")
        elif 'error' not in gamma_data:
            signal_components.append("Negative Gamma (Trending)")
        else:
            signal_components.append("Gamma data unavailable")
        
        # Final interpretation
        if signal_score > 0.3:
            direction = "BULLISH"
            confidence = "HIGH"
        elif signal_score > 0.1:
            direction = "BULLISH"
            confidence = "MEDIUM"
        elif signal_score > -0.1:
            direction = "NEUTRAL"
            confidence = "LOW"
        elif signal_score > -0.3:
            direction = "BEARISH"
            confidence = "MEDIUM"
        else:
            direction = "BEARISH"
            confidence = "HIGH"
        
        return {
            'timestamp': target_time,
            'spot_estimate': spot_price,
            'direction': direction,
            'confidence': confidence,
            'signal_score': signal_score,
            'components': signal_components,
            'detailed_metrics': {
                'pcr': pcr_data,
                'max_pain': max_pain,
                'support_resistance': levels,
                'gamma': gamma_data,
                'flows': flow_data,
                'iv_metrics': iv_data
            }
        }
    
    def _interpret_pcr(self, pcr_oi, pcr_volume):
        """Helper function to interpret PCR values"""
        if pcr_oi < 0.7 and pcr_volume < 0.8:
            return "Strong Bullish"
        elif pcr_oi < 0.9 and pcr_volume < 1.0:
            return "Mild Bullish"
        elif pcr_oi > 1.3 and pcr_volume > 1.2:
            return "Strong Bearish"
        elif pcr_oi > 1.1 and pcr_volume > 1.0:
            return "Mild Bearish"
        else:
            return "Neutral"
    
    def get_intraday_momentum(self, target_time="13:30:00", lookback_periods=5):
        """
        Calculate momentum based on recent PCR and flow changes
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        target_date = self.current_date
        target_datetime = pd.to_datetime(f"{target_date} {target_time}")
        
        # Get last few snapshots
        recent_snapshots = []
        for i in range(lookback_periods):
            check_time = target_datetime - timedelta(minutes=i*5)  # Every 5 minutes
            snapshot = self.get_snapshot_at_time(check_time.strftime("%H:%M:%S"), target_date)
            if len(snapshot) > 0:
                pcr_data = self.calculate_pcr_indicators(snapshot)
                recent_snapshots.append({
                    'time': check_time,
                    'pcr_oi': pcr_data.get('pcr_oi', 1),
                    'pcr_volume': pcr_data.get('pcr_volume', 1)
                })
        
        if len(recent_snapshots) < 3:
            return {'momentum': 'Insufficient data'}
        
        # Calculate PCR momentum
        pcr_trend = recent_snapshots[0]['pcr_oi'] - recent_snapshots[-1]['pcr_oi']
        
        if pcr_trend > 0.1:
            momentum = "Increasing Put bias (Bearish momentum)"
        elif pcr_trend < -0.1:
            momentum = "Decreasing Put bias (Bullish momentum)"
        else:
            momentum = "Sideways momentum"
        
        return {
            'momentum': momentum,
            'pcr_trend': pcr_trend,
            'recent_pcr_values': [s['pcr_oi'] for s in recent_snapshots]
        }
    
    def detect_unusual_activity(self, snapshot):
        """
        Detect unusual option activity
        """
        if len(snapshot) == 0:
            return {'error': 'Empty snapshot'}
        
        current_expiry = snapshot['Expiry_Date'].min()
        current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
        
        # Calculate volume-to-OI ratios
        current_exp_data['Call_Vol_OI_Ratio'] = current_exp_data['CALLS_Volume'] / (current_exp_data['CALLS_OI'] + 1)
        current_exp_data['Put_Vol_OI_Ratio'] = current_exp_data['PUTS_Volume'] / (current_exp_data['PUTS_OI'] + 1)
        
        # Identify strikes with unusual activity
        call_threshold = current_exp_data['Call_Vol_OI_Ratio'].quantile(0.8)
        put_threshold = current_exp_data['Put_Vol_OI_Ratio'].quantile(0.8)
        
        unusual_call_activity = current_exp_data[current_exp_data['Call_Vol_OI_Ratio'] > call_threshold]
        unusual_put_activity = current_exp_data[current_exp_data['Put_Vol_OI_Ratio'] > put_threshold]
        
        return {
            'unusual_calls': unusual_call_activity[['Strike_Price', 'CALLS_Volume', 'CALLS_OI', 'Call_Vol_OI_Ratio']].head(3),
            'unusual_puts': unusual_put_activity[['Strike_Price', 'PUTS_Volume', 'PUTS_OI', 'Put_Vol_OI_Ratio']].head(3)
        }
