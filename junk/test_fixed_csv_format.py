#!/usr/bin/env python3
"""
Test Fixed CSV Format
Shows the improved CSV format with proper columns and performance tracking.
"""

import pandas as pd
from datetime import datetime

# Create sample data with proper formatting
sample_data = [
    {
        'timestamp': '2025-08-25 10:00:00',
        'index': 'NIFTY',
        'atm_strike': 24800,
        'initial_spot_price': 24800,
        'signal_type': 'BULLISH',
        'confidence': 75.0,
        'pcr': 1.15,
        'atm_call_oi': 40123,
        'atm_put_oi': 27345,
        'atm_call_oi_change': 97187,
        'atm_put_oi_change': 12179,
        'suggested_trade': 'Buy Call (ATM/ITM) or Bull Call Spread',
        'current_spot_price': 24850,  # Updated 15 minutes later
        'current_option_premium': 0,  # Would be actual option price
        'price_change_percent': 0.20,  # (24850-24800)/24800 * 100
        'performance_status': 'PROFIT'  # BULLISH signal + price increase
    },
    {
        'timestamp': '2025-08-25 10:15:00',
        'index': 'NIFTY',
        'atm_strike': 24850,
        'initial_spot_price': 24850,
        'signal_type': 'NEUTRAL',
        'confidence': 50.0,
        'pcr': 1.05,
        'atm_call_oi': 40582,
        'atm_put_oi': 28162,
        'atm_call_oi_change': 12488,
        'atm_put_oi_change': 28480,
        'suggested_trade': 'Wait for clearer signal',
        'current_spot_price': 24850,  # Same as initial for new record
        'current_option_premium': 0,
        'price_change_percent': 0.0,
        'performance_status': 'ACTIVE'
    },
    {
        'timestamp': '2025-08-25 10:00:00',
        'index': 'BANKNIFTY',
        'atm_strike': 56000,
        'initial_spot_price': 56000,
        'signal_type': 'BEARISH',
        'confidence': 85.0,
        'pcr': 0.28,
        'atm_call_oi': 80938,
        'atm_put_oi': 27654,
        'atm_call_oi_change': 6780,
        'atm_put_oi_change': -76,
        'suggested_trade': 'Buy Put (ATM/ITM) or Bear Put Spread',
        'current_spot_price': 55800,  # Updated 15 minutes later
        'current_option_premium': 0,
        'price_change_percent': -0.36,  # (55800-56000)/56000 * 100
        'performance_status': 'PROFIT'  # BEARISH signal + price decrease
    }
]

# Create DataFrame
df = pd.DataFrame(sample_data)

# Save to CSV
csv_file = "results/options_tracker/sample_fixed_format.csv"
df.to_csv(csv_file, index=False)

print("✅ Sample Fixed CSV Format Created!")
print(f"📁 File: {csv_file}")
print("\n📊 Sample Data:")
print(df.to_string(index=False))

print("\n🔍 Key Improvements:")
print("1. ✅ Proper column separation (no concatenated values)")
print("2. ✅ Performance tracking columns added")
print("3. ✅ Current spot price tracking")
print("4. ✅ Price change percentage calculation")
print("5. ✅ Performance status (PROFIT/LOSS/NEUTRAL/ACTIVE)")

print("\n📈 Performance Tracking Example:")
print("- 10:00 AM: NIFTY BULLISH signal at ₹24,800")
print("- 10:15 AM: Price moved to ₹24,850 (+0.20%)")
print("- Status: PROFIT (BULLISH signal + price increase)")
print("- 10:00 AM: BANKNIFTY BEARISH signal at ₹56,000")
print("- 10:15 AM: Price moved to ₹55,800 (-0.36%)")
print("- Status: PROFIT (BEARISH signal + price decrease)")
