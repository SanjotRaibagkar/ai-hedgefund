#!/usr/bin/env python3
"""
Generate CSV files with stock predictions for up and down movements
"""

import sys
import os
import pandas as pd
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fno_rag.core.fno_engine import FNOEngine
    from fno_rag.models.data_models import HorizonType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Trying alternative import paths...")
    
    try:
        from src.fno_rag.core.fno_engine import FNOEngine
        from src.fno_rag.models.data_models import HorizonType
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("💡 Please ensure the FNO RAG system is properly installed")
        sys.exit(1)

def search_stocks_and_save_csv():
    """Search for stocks and save results to CSV files."""
    print("🚀 Initializing FNO Engine...")
    
    try:
        fno_engine = FNOEngine()
        print("✅ FNO Engine initialized successfully!")
        
        # Search 1: Stocks with high probability of moving up this week
        print("\n🔍 Searching for stocks with high probability of moving UP this week...")
        up_results = fno_engine.search_stocks(
            query="up week",
            horizon=HorizonType.WEEKLY,
            min_probability=0.3,
            max_results=15
        )
        
        print(f"Found {len(up_results)} stocks for UP movement:")
        up_data = []
        for i, result in enumerate(up_results, 1):
            print(f"{i}. {result.symbol}: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}, Neutral={result.neutral_probability:.1%}")
            up_data.append({
                'Symbol': result.symbol,
                'Up_Probability': result.up_probability,
                'Down_Probability': result.down_probability,
                'Neutral_Probability': result.neutral_probability,
                'Confidence_Score': getattr(result, 'confidence_score', 0.0),
                'Direction': 'UP',
                'Horizon': 'WEEKLY',
                'Query': 'up week'
            })
        
        # Search 2: Stocks with high probability of moving down this week
        print("\n🔍 Searching for stocks with high probability of moving DOWN this week...")
        down_results = fno_engine.search_stocks(
            query="down week",
            horizon=HorizonType.WEEKLY,
            min_probability=0.3,
            max_results=10
        )
        
        print(f"Found {len(down_results)} stocks for DOWN movement:")
        down_data = []
        for i, result in enumerate(down_results, 1):
            print(f"{i}. {result.symbol}: Up={result.up_probability:.1%}, Down={result.down_probability:.1%}, Neutral={result.neutral_probability:.1%}")
            down_data.append({
                'Symbol': result.symbol,
                'Up_Probability': result.up_probability,
                'Down_Probability': result.down_probability,
                'Neutral_Probability': result.neutral_probability,
                'Confidence_Score': getattr(result, 'confidence_score', 0.0),
                'Direction': 'DOWN',
                'Horizon': 'WEEKLY',
                'Query': 'down week'
            })
        
        # Combine all data
        all_data = up_data + down_data
        
        # Create DataFrame and save to CSV
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save combined results
            combined_filename = f"stock_predictions_combined_{timestamp}.csv"
            df.to_csv(combined_filename, index=False)
            print(f"\n✅ Combined results saved to: {combined_filename}")
            
            # Save separate files
            if up_data:
                up_df = pd.DataFrame(up_data)
                up_filename = f"stock_predictions_up_week_{timestamp}.csv"
                up_df.to_csv(up_filename, index=False)
                print(f"✅ UP predictions saved to: {up_filename}")
            
            if down_data:
                down_df = pd.DataFrame(down_data)
                down_filename = f"stock_predictions_down_week_{timestamp}.csv"
                down_df.to_csv(down_filename, index=False)
                print(f"✅ DOWN predictions saved to: {down_filename}")
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"   Total stocks analyzed: {len(all_data)}")
            print(f"   UP predictions: {len(up_data)}")
            print(f"   DOWN predictions: {len(down_data)}")
            
            return True
        else:
            print("❌ No results found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    print("📈 Stock Prediction CSV Generator")
    print("=" * 50)
    print(f"📅 Started at: {datetime.now()}")
    print("=" * 50)
    
    success = search_stocks_and_save_csv()
    
    if success:
        print("\n🎉 CSV generation completed successfully!")
    else:
        print("\n❌ CSV generation failed!")
    
    print(f"\n📅 Completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
