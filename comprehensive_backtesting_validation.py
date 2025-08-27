#!/usr/bin/env python3
"""
Comprehensive Backtesting Validation for FNO ML and RAG System
Tests predictions against actual results for the last 6 months
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import random
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fno_rag.core.fno_engine import FNOEngine
    from fno_rag.models.data_models import HorizonType, PredictionRequest
    from fno_rag.core.data_processor import FNODataProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Trying alternative import paths...")
    
    try:
        from src.fno_rag.core.fno_engine import FNOEngine
        from src.fno_rag.models.data_models import HorizonType, PredictionRequest
        from src.fno_rag.core.data_processor import FNODataProcessor
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("💡 Please ensure the FNO RAG system is properly installed")
        sys.exit(1)

class ComprehensiveBacktester:
    """Comprehensive backtesting system for FNO ML and RAG predictions."""
    
    def __init__(self):
        """Initialize the backtester."""
        self.fno_engine = None
        self.data_processor = None
        self.results = []
        
    def initialize_system(self):
        """Initialize the FNO engine and data processor."""
        print("🚀 Initializing FNO Engine for backtesting...")
        try:
            self.fno_engine = FNOEngine()
            self.data_processor = FNODataProcessor()
            print("✅ FNO Engine initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize FNO Engine: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available FNO symbols from the database."""
        try:
            # Get unique symbols from the database
            query = """
                SELECT DISTINCT TckrSymb as symbol
                FROM fno_bhav_copy
                WHERE TckrSymb IN ('NIFTY', 'BANKNIFTY', 'FINNIFTY', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'SBIN', 'AXISBANK')
                ORDER BY TckrSymb
            """
            
            # Use a fresh connection
            import duckdb
            db_path = "data/comprehensive_equity.duckdb"
            with duckdb.connect(db_path) as conn:
                result = conn.execute(query).fetchdf()
            
            symbols = result['symbol'].tolist() if not result.empty else ['NIFTY', 'BANKNIFTY']
            print(f"📊 Available symbols for backtesting: {symbols}")
            return symbols
            
        except Exception as e:
            print(f"⚠️ Error getting symbols, using defaults: {e}")
            return ['NIFTY', 'BANKNIFTY']
    
    def get_historical_data_range(self) -> Tuple[str, str]:
        """Get the date range for the last 6 months of data."""
        try:
            # Get the date range from the database
            query = """
                SELECT 
                    MIN(TRADE_DATE) as start_date,
                    MAX(TRADE_DATE) as end_date
                FROM fno_bhav_copy
            """
            
            # Use a fresh connection
            import duckdb
            db_path = "data/comprehensive_equity.duckdb"
            with duckdb.connect(db_path) as conn:
                result = conn.execute(query).fetchdf()
            
            if not result.empty:
                end_date = result.iloc[0]['end_date']
                start_date = end_date - timedelta(days=180)  # 6 months
                return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            else:
                # Fallback to current date - 6 months
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)
                return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                
        except Exception as e:
            print(f"⚠️ Error getting date range, using defaults: {e}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    def get_available_dates(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """Get available trading dates for a symbol."""
        try:
            query = """
                SELECT DISTINCT TRADE_DATE
                FROM fno_bhav_copy
                WHERE TckrSymb = ? 
                AND TRADE_DATE BETWEEN ? AND ?
                ORDER BY TRADE_DATE
            """
            
            # Use a fresh connection
            import duckdb
            db_path = "data/comprehensive_equity.duckdb"
            with duckdb.connect(db_path) as conn:
                result = conn.execute(query, [symbol, start_date, end_date]).fetchdf()
            
            dates = [row['TRADE_DATE'].strftime('%Y-%m-%d') for _, row in result.iterrows()]
            return dates
            
        except Exception as e:
            print(f"⚠️ Error getting dates for {symbol}: {e}")
            return []
    
    def get_actual_movement(self, symbol: str, prediction_date: str, horizon: HorizonType) -> Dict[str, Any]:
        """Get actual price movement for a symbol after the prediction date."""
        try:
            # Calculate the target date based on horizon
            pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
            
            if horizon == HorizonType.DAILY:
                target_date = pred_date + timedelta(days=1)
            elif horizon == HorizonType.WEEKLY:
                target_date = pred_date + timedelta(days=7)
            elif horizon == HorizonType.MONTHLY:
                target_date = pred_date + timedelta(days=30)
            else:
                target_date = pred_date + timedelta(days=1)
            
            # Get price data for prediction date and target date
            query = """
                SELECT 
                    TRADE_DATE,
                    ClsPric as close_price
                FROM fno_bhav_copy
                WHERE TckrSymb = ? 
                AND TRADE_DATE IN (?, ?)
                ORDER BY TRADE_DATE
            """
            
            # Use a fresh connection
            import duckdb
            db_path = "data/comprehensive_equity.duckdb"
            with duckdb.connect(db_path) as conn:
                result = conn.execute(query, [symbol, prediction_date, target_date.strftime('%Y-%m-%d')]).fetchdf()
            
            if len(result) >= 2:
                pred_price = result.iloc[0]['close_price']
                target_price = result.iloc[1]['close_price']
                
                # Calculate percentage change
                pct_change = ((target_price - pred_price) / pred_price) * 100
                
                # Determine actual direction
                if pct_change > 2:  # More than 2% up
                    actual_direction = 'UP'
                elif pct_change < -2:  # More than 2% down
                    actual_direction = 'DOWN'
                else:
                    actual_direction = 'NEUTRAL'
                
                return {
                    'prediction_date': prediction_date,
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'prediction_price': pred_price,
                    'target_price': target_price,
                    'actual_pct_change': pct_change,
                    'actual_direction': actual_direction,
                    'horizon_days': (target_date - pred_date).days
                }
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Error getting actual movement for {symbol} on {prediction_date}: {e}")
            return None
    
    def run_prediction_backtest(self, symbol: str, prediction_date: str) -> Dict[str, Any]:
        """Run backtest for a specific symbol and date."""
        try:
            print(f"  🔍 Testing {symbol} on {prediction_date}...")
            
            # Get predictions for all horizons
            predictions = {}
            for horizon in [HorizonType.DAILY, HorizonType.WEEKLY, HorizonType.MONTHLY]:
                try:
                    # Create prediction request
                    request = PredictionRequest(
                        symbol=symbol,
                        horizon=horizon,
                        include_explanations=False,
                        top_k_similar=3
                    )
                    
                    # Get prediction
                    result = self.fno_engine.predict_probability(request)
                    
                    predictions[horizon.value] = {
                        'up_probability': result.up_probability,
                        'down_probability': result.down_probability,
                        'neutral_probability': result.neutral_probability,
                        'confidence_score': result.confidence_score,
                        'predicted_direction': 'UP' if result.up_probability > result.down_probability else 'DOWN'
                    }
                    
                except Exception as e:
                    print(f"    ⚠️ Error predicting {symbol} {horizon.value}: {e}")
                    predictions[horizon.value] = {
                        'up_probability': 0.33,
                        'down_probability': 0.33,
                        'neutral_probability': 0.34,
                        'confidence_score': 0.5,
                        'predicted_direction': 'NEUTRAL'
                    }
            
            # Get actual movements
            actual_movements = {}
            for horizon in [HorizonType.DAILY, HorizonType.WEEKLY, HorizonType.MONTHLY]:
                actual = self.get_actual_movement(symbol, prediction_date, horizon)
                if actual:
                    actual_movements[horizon.value] = actual
                else:
                    actual_movements[horizon.value] = None
            
            # Calculate accuracy metrics
            accuracy_metrics = {}
            for horizon in ['daily', 'weekly', 'monthly']:
                if horizon in predictions and horizon in actual_movements and actual_movements[horizon]:
                    pred_direction = predictions[horizon]['predicted_direction']
                    actual_direction = actual_movements[horizon]['actual_direction']
                    
                    # Check if prediction was correct
                    is_correct = pred_direction == actual_direction
                    
                    # Calculate probability accuracy (how confident the model was in the correct direction)
                    if actual_direction == 'UP':
                        prob_accuracy = predictions[horizon]['up_probability']
                    elif actual_direction == 'DOWN':
                        prob_accuracy = predictions[horizon]['down_probability']
                    else:
                        prob_accuracy = predictions[horizon]['neutral_probability']
                    
                    accuracy_metrics[horizon] = {
                        'is_correct': is_correct,
                        'prob_accuracy': prob_accuracy,
                        'confidence_score': predictions[horizon]['confidence_score']
                    }
                else:
                    accuracy_metrics[horizon] = {
                        'is_correct': False,
                        'prob_accuracy': 0.0,
                        'confidence_score': 0.0
                    }
            
            return {
                'symbol': symbol,
                'prediction_date': prediction_date,
                'predictions': predictions,
                'actual_movements': actual_movements,
                'accuracy_metrics': accuracy_metrics
            }
            
        except Exception as e:
            print(f"  ❌ Error in backtest for {symbol} on {prediction_date}: {e}")
            return None
    
    def run_comprehensive_backtest(self, num_test_days: int = 10) -> pd.DataFrame:
        """Run comprehensive backtesting for multiple symbols and dates."""
        print("🎯 Starting Comprehensive Backtesting")
        print("=" * 60)
        
        # Initialize system
        if not self.initialize_system():
            return pd.DataFrame()
        
        # Get available symbols and date range
        symbols = self.get_available_symbols()
        start_date, end_date = self.get_historical_data_range()
        
        print(f"📅 Testing period: {start_date} to {end_date}")
        print(f"📊 Testing symbols: {symbols}")
        print(f"🎲 Number of test days: {num_test_days}")
        
        all_results = []
        
        # For each symbol, select random test dates
        for symbol in symbols:
            print(f"\n📈 Testing {symbol}...")
            
            # Get available dates for this symbol
            available_dates = self.get_available_dates(symbol, start_date, end_date)
            
            if len(available_dates) < num_test_days:
                test_dates = available_dates
                print(f"  ⚠️ Only {len(available_dates)} dates available for {symbol}")
            else:
                # Select random test dates
                test_dates = random.sample(available_dates, num_test_days)
            
            # Run backtest for each test date
            for test_date in test_dates:
                result = self.run_prediction_backtest(symbol, test_date)
                if result:
                    all_results.append(result)
        
        # Convert results to DataFrame
        df_results = self.convert_results_to_dataframe(all_results)
        
        return df_results
    
    def convert_results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert backtest results to a comprehensive DataFrame."""
        rows = []
        
        for result in results:
            symbol = result['symbol']
            prediction_date = result['prediction_date']
            
            for horizon in ['daily', 'weekly', 'monthly']:
                predictions = result['predictions'].get(horizon, {})
                actual_movements = result['actual_movements'].get(horizon, {})
                accuracy_metrics = result['accuracy_metrics'].get(horizon, {})
                
                row = {
                    'Symbol': symbol,
                    'Prediction_Date': prediction_date,
                    'Horizon': horizon.upper(),
                    
                    # Prediction details
                    'Predicted_Up_Prob': predictions.get('up_probability', 0.0),
                    'Predicted_Down_Prob': predictions.get('down_probability', 0.0),
                    'Predicted_Neutral_Prob': predictions.get('neutral_probability', 0.0),
                    'Predicted_Direction': predictions.get('predicted_direction', 'NEUTRAL'),
                    'Confidence_Score': predictions.get('confidence_score', 0.0),
                    
                    # Actual results
                    'Target_Date': actual_movements.get('target_date', 'N/A') if actual_movements else 'N/A',
                    'Prediction_Price': actual_movements.get('prediction_price', 0.0) if actual_movements else 0.0,
                    'Target_Price': actual_movements.get('target_price', 0.0) if actual_movements else 0.0,
                    'Actual_Pct_Change': actual_movements.get('actual_pct_change', 0.0) if actual_movements else 0.0,
                    'Actual_Direction': actual_movements.get('actual_direction', 'N/A') if actual_movements else 'N/A',
                    'Horizon_Days': actual_movements.get('horizon_days', 0) if actual_movements else 0,
                    
                    # Accuracy metrics
                    'Prediction_Correct': accuracy_metrics.get('is_correct', False),
                    'Probability_Accuracy': accuracy_metrics.get('prob_accuracy', 0.0),
                    'Model_Confidence': accuracy_metrics.get('confidence_score', 0.0)
                }
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        if df.empty:
            return {}
        
        # Overall accuracy by horizon
        horizon_accuracy = {}
        for horizon in ['DAILY', 'WEEKLY', 'MONTHLY']:
            horizon_data = df[df['Horizon'] == horizon]
            if not horizon_data.empty:
                correct_predictions = horizon_data['Prediction_Correct'].sum()
                total_predictions = len(horizon_data)
                accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
                
                avg_confidence = horizon_data['Model_Confidence'].mean()
                avg_prob_accuracy = horizon_data['Probability_Accuracy'].mean()
                
                horizon_accuracy[horizon] = {
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions,
                    'accuracy_percentage': accuracy,
                    'avg_confidence': avg_confidence,
                    'avg_probability_accuracy': avg_prob_accuracy
                }
        
        # Symbol-wise accuracy
        symbol_accuracy = {}
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]
            correct_predictions = symbol_data['Prediction_Correct'].sum()
            total_predictions = len(symbol_data)
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            
            symbol_accuracy[symbol] = {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy_percentage': accuracy
            }
        
        # Overall statistics
        total_predictions = len(df)
        total_correct = df['Prediction_Correct'].sum()
        overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
        
        avg_confidence = df['Model_Confidence'].mean()
        avg_prob_accuracy = df['Probability_Accuracy'].mean()
        
        return {
            'overall_stats': {
                'total_predictions': total_predictions,
                'correct_predictions': total_correct,
                'accuracy_percentage': overall_accuracy,
                'avg_confidence': avg_confidence,
                'avg_probability_accuracy': avg_prob_accuracy
            },
            'horizon_accuracy': horizon_accuracy,
            'symbol_accuracy': symbol_accuracy
        }

def main():
    """Main function to run comprehensive backtesting."""
    print("🎯 Comprehensive FNO ML and RAG Backtesting")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now()}")
    print("=" * 60)
    
    # Initialize backtester
    backtester = ComprehensiveBacktester()
    
    # Run comprehensive backtest
    df_results = backtester.run_comprehensive_backtest(num_test_days=10)
    
    if not df_results.empty:
        # Generate summary report
        summary = backtester.generate_summary_report(df_results)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"comprehensive_backtest_results_{timestamp}.csv"
        df_results.to_csv(csv_filename, index=False)
        
        # Print summary
        print(f"\n📊 Backtesting Results Summary")
        print("=" * 60)
        
        if 'overall_stats' in summary:
            stats = summary['overall_stats']
            print(f"📈 Overall Accuracy: {stats['accuracy_percentage']:.2f}% ({stats['correct_predictions']}/{stats['total_predictions']})")
            print(f"🎯 Average Confidence: {stats['avg_confidence']:.2f}")
            print(f"📊 Average Probability Accuracy: {stats['avg_probability_accuracy']:.2f}")
        
        print(f"\n📅 Horizon-wise Accuracy:")
        for horizon, acc in summary.get('horizon_accuracy', {}).items():
            print(f"  {horizon}: {acc['accuracy_percentage']:.2f}% ({acc['correct_predictions']}/{acc['total_predictions']})")
        
        print(f"\n🏢 Symbol-wise Accuracy:")
        for symbol, acc in summary.get('symbol_accuracy', {}).items():
            print(f"  {symbol}: {acc['accuracy_percentage']:.2f}% ({acc['correct_predictions']}/{acc['total_predictions']})")
        
        print(f"\n✅ Results saved to: {csv_filename}")
        print(f"📊 Total predictions tested: {len(df_results)}")
        
    else:
        print("❌ No results generated from backtesting")
    
    print(f"\n📅 Completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
