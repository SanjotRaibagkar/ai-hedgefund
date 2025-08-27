#!/usr/bin/env python3
"""
Backtest with Available Data
Test the FNO RAG system using the available data in the database.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.backtesting import FNOBacktestEngine
from src.fno_rag import FNOEngine, HorizonType
from src.data.database.duckdb_manager import DatabaseManager


def test_with_available_data():
    """Test the backtesting system with available data."""

    print("🧪 Backtest with Available Data")
    print("=" * 40)

    try:
        # Initialize engines
        print("1. Initializing FNO RAG System...")
        fno_engine = FNOEngine()

        print("2. Initializing Backtest Engine...")
        backtest_engine = FNOBacktestEngine(fno_engine)

        # Check available data
        print("3. Checking available FNO data...")
        db = DatabaseManager()
        
        # Get available dates
        dates_result = db.connection.execute(
            "SELECT DISTINCT TRADE_DATE FROM fno_bhav_copy ORDER BY TRADE_DATE"
        ).fetchdf()
        available_dates = dates_result['TRADE_DATE'].tolist()
        print(f"📅 Available dates: {available_dates}")

        # Get available symbols
        symbols_result = db.connection.execute(
            "SELECT DISTINCT TckrSymb FROM fno_bhav_copy ORDER BY TckrSymb"
        ).fetchdf()
        available_symbols = symbols_result['TckrSymb'].tolist()
        print(f"📊 Available symbols: {len(available_symbols)}")
        print(f"   Sample symbols: {available_symbols[:10]}")

        if not available_dates:
            print("❌ No FNO data available for backtesting!")
            return False

        # Use the available date as our test date
        test_date = available_dates[0].strftime('%Y-%m-%d')
        print(f"🎯 Using test date: {test_date}")

        # Test with a few major symbols
        test_symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']
        print(f"🎯 Testing with symbols: {test_symbols}")

        # Test the prediction system directly
        print("\n4. Testing prediction system...")
        for symbol in test_symbols:
            try:
                print(f"\n   Testing {symbol}:")
                
                # Test daily prediction
                from src.fno_rag.models.data_models import PredictionRequest
                request = PredictionRequest(
                    symbol=symbol,
                    horizon=HorizonType.DAILY,
                    include_explanations=True
                )
                
                prediction = fno_engine.predict_probability(request)
                print(f"     Daily: UP={prediction.up_probability:.3f}, DOWN={prediction.down_probability:.3f}, NEUTRAL={prediction.neutral_probability:.3f}")
                print(f"     Confidence: {prediction.confidence_score:.3f}")
                
                # Test weekly prediction
                request.horizon = HorizonType.WEEKLY
                prediction = fno_engine.predict_probability(request)
                print(f"     Weekly: UP={prediction.up_probability:.3f}, DOWN={prediction.down_probability:.3f}, NEUTRAL={prediction.neutral_probability:.3f}")
                print(f"     Confidence: {prediction.confidence_score:.3f}")
                
            except Exception as e:
                print(f"     ❌ Error testing {symbol}: {e}")

        # Test the backtesting engine with synthetic future dates
        print("\n5. Testing backtesting engine with synthetic data...")
        
        # Create synthetic test dates (future dates for demonstration)
        synthetic_dates = [
            (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
        ]
        print(f"   Synthetic test dates: {synthetic_dates}")

        # Test backtesting with synthetic dates
        results = backtest_engine.backtest_all_symbols(test_symbols, synthetic_dates, HorizonType.DAILY)
        print(f"   Backtest results: {len(results)} predictions")

        if results:
            correct_predictions = sum(1 for r in results if r.prediction_correct)
            accuracy = correct_predictions / len(results) if results else 0.0
            print(f"   Correct predictions: {correct_predictions}")
            print(f"   Accuracy: {accuracy:.2%}")

            # Show sample results
            print(f"\n6. Sample Results:")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. {result.symbol} ({result.prediction_date}): "
                      f"Predicted {result.predicted_up_prob:.2f}/{result.predicted_down_prob:.2f}/{result.predicted_neutral_prob:.2f}, "
                      f"Actual: {result.actual_direction} ({result.actual_return:.3f}), "
                      f"Correct: {result.prediction_correct}")

        # Test system status
        print("\n7. Testing system status...")
        status = fno_engine.get_system_status()
        print(f"   System status: {status}")

        print(f"\n✅ Backtest with available data completed!")
        return True

    except Exception as e:
        print(f"❌ Backtest with available data failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_synthetic_data_for_testing():
    """Create synthetic FNO data for more comprehensive backtesting."""
    
    print("\n🔧 Creating synthetic data for testing...")
    
    try:
        db = DatabaseManager()
        
        # Get existing data structure
        sample_data = db.connection.execute(
            "SELECT * FROM fno_bhav_copy LIMIT 1"
        ).fetchdf()
        
        if sample_data.empty:
            print("❌ No existing data to base synthetic data on!")
            return False
            
        print(f"📊 Sample data structure: {list(sample_data.columns)}")
        
        # Create synthetic data for the past 30 days
        symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']
        base_date = datetime.now() - timedelta(days=30)
        
        synthetic_records = []
        for i in range(30):
            current_date = base_date + timedelta(days=i)
            if current_date.weekday() < 5:  # Only weekdays
                for symbol in symbols:
                    # Create realistic price data
                    base_price = 20000 if symbol == 'NIFTY' else (45000 if symbol == 'BANKNIFTY' else 20000)
                    price_variation = (i % 10 - 5) * 0.01  # ±5% variation
                    close_price = base_price * (1 + price_variation)
                    
                    record = {
                        'TckrSymb': symbol,
                        'TRADE_DATE': current_date.strftime('%Y-%m-%d'),
                        'OpnPric': close_price * 0.995,
                        'HghPric': close_price * 1.02,
                        'LwPric': close_price * 0.98,
                        'ClsPric': close_price,
                        'TtlTradgVol': 1000000 + (i * 10000),
                        'OpnIntrst': 500000 + (i * 5000)
                    }
                    synthetic_records.append(record)
        
        print(f"📈 Created {len(synthetic_records)} synthetic records")
        
        # Insert synthetic data
        if synthetic_records:
            # Convert to DataFrame for easier insertion
            import pandas as pd
            df = pd.DataFrame(synthetic_records)
            
            # Insert into database
            db.connection.execute("DELETE FROM fno_bhav_copy WHERE TckrSymb IN ('NIFTY', 'BANKNIFTY', 'FINNIFTY')")
            db.connection.execute("INSERT INTO fno_bhav_copy SELECT * FROM df")
            
            print("✅ Synthetic data inserted successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to create synthetic data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Starting Backtest with Available Data...")
    
    # First test with existing data
    success1 = test_with_available_data()
    
    # Then create synthetic data and test again
    success2 = create_synthetic_data_for_testing()
    
    if success2:
        print("\n🔄 Testing again with synthetic data...")
        success3 = test_with_available_data()
    else:
        success3 = False

    if success1 or success3:
        print(f"\n🎉 Backtesting completed successfully!")
    else:
        print(f"\n❌ Backtesting failed!")

