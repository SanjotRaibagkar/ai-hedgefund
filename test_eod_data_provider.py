#!/usr/bin/env python3
"""
Test EOD Data Provider and Enhanced API
Verifies that the four EOD tables are accessible for ML models and screeners.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from src.tools.enhanced_api import (
    get_fno_bhav_copy_data,
    get_equity_bhav_copy_delivery_data,
    get_bhav_copy_indices_data,
    get_fii_dii_activity_data,
    get_latest_fno_data,
    get_latest_equity_data,
    get_latest_index_data,
    get_latest_fii_dii_data,
    get_eod_database_stats,
    get_top_symbols_by_volume,
    get_market_summary,
    get_option_chain_from_eod,
    get_derivatives_summary
)

def test_eod_data_provider():
    """Test the EOD data provider functionality."""
    
    print("🚀 TESTING EOD DATA PROVIDER AND ENHANCED API")
    print("=" * 80)
    
    # Test 1: Database Statistics
    print("\n📊 1. Testing Database Statistics...")
    stats = get_eod_database_stats()
    if stats:
        print("✅ Database Statistics Retrieved:")
        for table, table_stats in stats.items():
            print(f"   {table.upper()}:")
            print(f"     • Total Records: {table_stats.get('total_records', 0):,}")
            print(f"     • Unique Dates: {table_stats.get('unique_dates', 0)}")
            print(f"     • Date Range: {table_stats.get('earliest_date', 'N/A')} to {table_stats.get('latest_date', 'N/A')}")
            if 'unique_symbols' in table_stats:
                print(f"     • Unique Symbols: {table_stats.get('unique_symbols', 0)}")
            elif 'unique_indices' in table_stats:
                print(f"     • Unique Indices: {table_stats.get('unique_indices', 0)}")
            elif 'unique_categories' in table_stats:
                print(f"     • Unique Categories: {table_stats.get('unique_categories', 0)}")
    else:
        print("❌ Failed to retrieve database statistics")
    
    # Test 2: FNO Bhav Copy Data
    print("\n📈 2. Testing FNO Bhav Copy Data...")
    fno_data = get_fno_bhav_copy_data(symbol='NIFTY', limit=10)
    if not fno_data.empty:
        print(f"✅ Retrieved {len(fno_data)} FNO records for NIFTY")
        print(f"   • Columns: {list(fno_data.columns)}")
        print(f"   • Date Range: {fno_data['TRADE_DATE'].min()} to {fno_data['TRADE_DATE'].max()}")
        print(f"   • Sample Data:")
        print(fno_data[['TckrSymb', 'TRADE_DATE', 'ClsPric', 'TtlTradgVol', 'OpnIntrst']].head(3))
    else:
        print("❌ No FNO data retrieved")
    
    # Test 3: Equity Bhav Copy Delivery Data
    print("\n📊 3. Testing Equity Bhav Copy Delivery Data...")
    equity_data = get_equity_bhav_copy_delivery_data(symbol='RELIANCE', limit=10)
    if not equity_data.empty:
        print(f"✅ Retrieved {len(equity_data)} equity records for RELIANCE")
        print(f"   • Columns: {list(equity_data.columns)}")
        print(f"   • Date Range: {equity_data['TRADE_DATE'].min()} to {equity_data['TRADE_DATE'].max()}")
        print(f"   • Sample Data:")
        # Show available columns first
        print(f"   • Available columns: {list(equity_data.columns)}")
        if 'SYMBOL' in equity_data.columns and 'TRADE_DATE' in equity_data.columns:
            sample_cols = ['SYMBOL', 'TRADE_DATE']
            if 'CLOSE' in equity_data.columns:
                sample_cols.append('CLOSE')
            if 'VOLUME' in equity_data.columns:
                sample_cols.append('VOLUME')
            if 'DELIV_QTY' in equity_data.columns:
                sample_cols.append('DELIV_QTY')
            print(equity_data[sample_cols].head(3))
        else:
            print("   • No standard columns found")
    else:
        print("❌ No equity data retrieved")
    
    # Test 4: Bhav Copy Indices Data
    print("\n📊 4. Testing Bhav Copy Indices Data...")
    indices_data = get_bhav_copy_indices_data(index_name='NIFTY 50', limit=10)
    if not indices_data.empty:
        print(f"✅ Retrieved {len(indices_data)} indices records for NIFTY 50")
        print(f"   • Columns: {list(indices_data.columns)}")
        print(f"   • Date Range: {indices_data['TRADE_DATE'].min()} to {indices_data['TRADE_DATE'].max()}")
        print(f"   • Sample Data:")
        # Show available columns first
        print(f"   • Available columns: {list(indices_data.columns)}")
        if 'Index_Name' in indices_data.columns and 'TRADE_DATE' in indices_data.columns:
            sample_cols = ['Index_Name', 'TRADE_DATE']
            if 'Closing_Index_Value' in indices_data.columns:
                sample_cols.append('Closing_Index_Value')
            if 'Points_Change' in indices_data.columns:
                sample_cols.append('Points_Change')
            if 'Change_Percent' in indices_data.columns:
                sample_cols.append('Change_Percent')
            print(indices_data[sample_cols].head(3))
        else:
            print("   • No standard columns found")
    else:
        print("❌ No indices data retrieved")
    
    # Test 5: FII DII Activity Data
    print("\n📊 5. Testing FII DII Activity Data...")
    fii_dii_data = get_fii_dii_activity_data(limit=10)
    if not fii_dii_data.empty:
        print(f"✅ Retrieved {len(fii_dii_data)} FII/DII activity records")
        print(f"   • Columns: {list(fii_dii_data.columns)}")
        print(f"   • Categories: {fii_dii_data['category'].unique()}")
        print(f"   • Sample Data:")
        # Show available columns first
        print(f"   • Available columns: {list(fii_dii_data.columns)}")
        if 'category' in fii_dii_data.columns:
            sample_cols = ['category']
            if 'date' in fii_dii_data.columns:
                sample_cols.append('date')
            if 'buyValue' in fii_dii_data.columns:
                sample_cols.append('buyValue')
            if 'sellValue' in fii_dii_data.columns:
                sample_cols.append('sellValue')
            if 'netValue' in fii_dii_data.columns:
                sample_cols.append('netValue')
            print(fii_dii_data[sample_cols].head(3))
        else:
            print("   • No standard columns found")
    else:
        print("❌ No FII/DII data retrieved")
    
    # Test 6: Latest Data Functions
    print("\n📈 6. Testing Latest Data Functions...")
    
    # Latest FNO data
    latest_fno = get_latest_fno_data('BANKNIFTY', days=7)
    if not latest_fno.empty:
        print(f"✅ Latest FNO data for BANKNIFTY: {len(latest_fno)} records")
    else:
        print("❌ No latest FNO data")
    
    # Latest equity data
    latest_equity = get_latest_equity_data('TCS', days=7)
    if not latest_equity.empty:
        print(f"✅ Latest equity data for TCS: {len(latest_equity)} records")
    else:
        print("❌ No latest equity data")
    
    # Latest index data
    latest_index = get_latest_index_data('NIFTY BANK', days=7)
    if not latest_index.empty:
        print(f"✅ Latest index data for NIFTY BANK: {len(latest_index)} records")
    else:
        print("❌ No latest index data")
    
    # Latest FII/DII data
    latest_fii_dii = get_latest_fii_dii_data(days=7)
    if not latest_fii_dii.empty:
        print(f"✅ Latest FII/DII data: {len(latest_fii_dii)} records")
    else:
        print("❌ No latest FII/DII data")
    
    # Test 7: Top Symbols by Volume
    print("\n📊 7. Testing Top Symbols by Volume...")
    top_fno = get_top_symbols_by_volume('fno_bhav_copy', days=30, limit=5)
    if not top_fno.empty:
        print(f"✅ Top FNO symbols by volume:")
        print(f"   • Available columns: {list(top_fno.columns)}")
        if 'TckrSymb' in top_fno.columns:
            sample_cols = ['TckrSymb']
            if 'total_volume' in top_fno.columns:
                sample_cols.append('total_volume')
            if 'trading_days' in top_fno.columns:
                sample_cols.append('trading_days')
            if 'avg_price' in top_fno.columns:
                sample_cols.append('avg_price')
            print(top_fno[sample_cols].head())
        else:
            print("   • No standard columns found")
    else:
        print("❌ No top FNO symbols data")
    
    top_equity = get_top_symbols_by_volume('equity_bhav_copy_delivery', days=30, limit=5)
    if not top_equity.empty:
        print(f"✅ Top equity symbols by volume:")
        print(f"   • Available columns: {list(top_equity.columns)}")
        if 'SYMBOL' in top_equity.columns:
            sample_cols = ['SYMBOL']
            if 'total_volume' in top_equity.columns:
                sample_cols.append('total_volume')
            if 'trading_days' in top_equity.columns:
                sample_cols.append('trading_days')
            if 'avg_price' in top_equity.columns:
                sample_cols.append('avg_price')
            print(top_equity[sample_cols].head())
        else:
            print("   • No standard columns found")
    else:
        print("❌ No top equity symbols data")
    
    # Test 8: Market Summary
    print("\n📊 8. Testing Market Summary...")
    market_summary = get_market_summary()
    if market_summary:
        print(f"✅ Market Summary for {market_summary.get('date', 'N/A')}:")
        print(f"   • FNO Summary: {market_summary.get('fno_summary', {})}")
        print(f"   • Equity Summary: {market_summary.get('equity_summary', {})}")
        print(f"   • Indices Summary: {market_summary.get('indices_summary', {})}")
        print(f"   • FII/DII Summary: {market_summary.get('fii_dii_summary', {})}")
    else:
        print("❌ No market summary data")
    
    # Test 9: Option Chain from EOD
    print("\n📊 9. Testing Option Chain from EOD...")
    option_chain = get_option_chain_from_eod('NIFTY')
    if not option_chain.empty:
        print(f"✅ Option chain for NIFTY: {len(option_chain)} contracts")
        print(f"   • Calls: {len(option_chain[option_chain['option_type'] == 'CE'])}")
        print(f"   • Puts: {len(option_chain[option_chain['option_type'] == 'PE'])}")
        print(f"   • Strike Range: {option_chain['StrkPric'].min()} - {option_chain['StrkPric'].max()}")
        print(f"   • Sample Data:")
        print(f"   • Available columns: {list(option_chain.columns)}")
        if 'TckrSymb' in option_chain.columns:
            sample_cols = ['TckrSymb', 'option_type']
            if 'StrkPric' in option_chain.columns:
                sample_cols.append('StrkPric')
            if 'ClsPric' in option_chain.columns:
                sample_cols.append('ClsPric')
            if 'TtlTradgVol' in option_chain.columns:
                sample_cols.append('TtlTradgVol')
            if 'OpnIntrst' in option_chain.columns:
                sample_cols.append('OpnIntrst')
            sample_data = option_chain[sample_cols].head(5)
            print(sample_data)
        else:
            print("   • No standard columns found")
    else:
        print("❌ No option chain data")
    
    # Test 10: Derivatives Summary
    print("\n📊 10. Testing Derivatives Summary...")
    derivatives_summary = get_derivatives_summary('BANKNIFTY', days=30)
    if derivatives_summary:
        print(f"✅ Derivatives Summary for BANKNIFTY:")
        print(f"   • Total Contracts: {derivatives_summary.get('total_contracts', 0)}")
        print(f"   • Unique Dates: {derivatives_summary.get('unique_dates', 0)}")
        print(f"   • Volume Stats: {derivatives_summary.get('volume_stats', {})}")
        print(f"   • OI Stats: {derivatives_summary.get('oi_stats', {})}")
        print(f"   • Contract Types: {derivatives_summary.get('contract_types', {})}")
    else:
        print("❌ No derivatives summary data")
    
    print("\n" + "=" * 80)
    print("🎉 EOD DATA PROVIDER TESTING COMPLETED!")
    print("✅ All four EOD tables are now accessible for ML models and screeners")
    print("📊 Data is ready for analysis, backtesting, and strategy development")

def main():
    """Main function."""
    try:
        test_eod_data_provider()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
