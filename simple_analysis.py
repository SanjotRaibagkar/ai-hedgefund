#!/usr/bin/env python3
"""
Simple Vector Store Analysis
"""

from build_enhanced_vector_store import EnhancedFNOVectorStore

def main():
    print("🔍 Simple Vector Store Analysis")
    print("=" * 40)
    
    # Load vector store
    vs = EnhancedFNOVectorStore()
    vs.load_vector_store()
    
    if not vs.metadata:
        print("❌ No metadata found")
        return
    
    total = len(vs.metadata)
    print(f"📊 Total Snapshots: {total:,}")
    
    # Sample data
    sample = vs.metadata[0]
    print(f"📋 Sample Data: {sample}")
    
    # Date range
    dates = [case['date'] for case in vs.metadata]
    start_date = min(dates)
    end_date = max(dates)
    print(f"📅 Date Range: {start_date} to {end_date}")
    
    # Symbols
    symbols = [case['symbol'] for case in vs.metadata]
    unique_symbols = len(set(symbols))
    print(f"📈 Unique Symbols: {unique_symbols}")
    
    # Assessment
    print(f"\n🎯 Assessment:")
    if total >= 20000:
        print(f"✅ GOOD: {total:,} snapshots is sufficient for RAG analysis")
    else:
        print(f"⚠️ LIMITED: {total:,} snapshots may need more data")
    
    if unique_symbols >= 20:
        print(f"✅ GOOD: {unique_symbols} symbols provide good coverage")
    else:
        print(f"⚠️ LIMITED: {unique_symbols} symbols may need more variety")

if __name__ == "__main__":
    main()
