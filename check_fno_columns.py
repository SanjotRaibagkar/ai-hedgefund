#!/usr/bin/env python3
"""
Check FNO Bhav Copy Table Columns
"""

import duckdb

def check_fno_columns():
    """Check the columns and details of fno_bhav_copy table."""
    print("📊 FNO BHAV COPY TABLE COLUMN DETAILS")
    print("=" * 80)
    
    try:
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Get schema
        print("\n📋 TABLE SCHEMA:")
        schema = conn.execute('DESCRIBE fno_bhav_copy').fetchdf()
        
        for idx, row in schema.iterrows():
            print(f"{idx+1:2d}. {row['column_name']:<25} | {row['column_type']:<12} | Null: {row['null']} | Key: {row['key']}")
        
        # Get sample data
        print("\n📈 SAMPLE DATA:")
        sample = conn.execute('''
            SELECT FinInstrmId, TckrSymb, XpryDt, StrkPric, OptnTp, 
                   ClsPric, OpnIntrst, TtlTradgVol, TRADE_DATE 
            FROM fno_bhav_copy 
            LIMIT 5
        ''').fetchdf()
        
        print(sample)
        
        # Get data count
        count = conn.execute('SELECT COUNT(*) FROM fno_bhav_copy').fetchone()[0]
        print(f"\n📊 TOTAL RECORDS: {count:,}")
        
        # Get unique symbols
        symbols = conn.execute('SELECT COUNT(DISTINCT TckrSymb) FROM fno_bhav_copy').fetchone()[0]
        print(f"🎯 UNIQUE SYMBOLS: {symbols}")
        
        # Get date range
        date_range = conn.execute('SELECT MIN(TRADE_DATE), MAX(TRADE_DATE) FROM fno_bhav_copy').fetchone()
        print(f"📅 DATE RANGE: {date_range[0]} to {date_range[1]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_fno_columns()
