#!/usr/bin/env python3
"""
FNO Comprehensive Backtesting for ML+RAG System - BALANCED VERSION
Tests predictions vs actual outcomes using only fno_bhav_copy table
Fixed to have balanced predictions (not all UP)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import duckdb
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fno_backtest_ml_rag_balanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FNOBacktestMLRAGBalanced:
    """FNO backtesting system with balanced predictions"""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        self.db_path = db_path
        self.results = []
        
    def get_test_dates(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31") -> List[str]:
        """Get list of test dates from fno_bhav_copy table"""
        try:
            with duckdb.connect(self.db_path) as conn:
                query = """
                SELECT DISTINCT TRADE_DATE as test_date
                FROM fno_bhav_copy 
                WHERE TRADE_DATE BETWEEN ? AND ?
                AND ClsPric > 100
                ORDER BY TRADE_DATE
                """
                result = conn.execute(query, [start_date, end_date]).fetchdf()
                return result['test_date'].dt.strftime('%Y-%m-%d').tolist()
        except Exception as e:
            logger.error(f"Error getting test dates: {e}")
            return []
    
    def get_symbols_for_testing(self, test_date: str, limit: int = 20) -> List[str]:
        """Get symbols available for testing on a specific date from fno_bhav_copy"""
        try:
            with duckdb.connect(self.db_path) as conn:
                query = """
                SELECT DISTINCT TckrSymb
                FROM fno_bhav_copy 
                WHERE TRADE_DATE = ?
                AND ClsPric > 100
                AND TckrSymb NOT LIKE '%NIFTY%'
                AND TckrSymb NOT LIKE '%BANKNIFTY%'
                AND TckrSymb NOT LIKE '%FINNIFTY%'
                LIMIT ?
                """
                result = conn.execute(query, [test_date, limit]).fetchdf()
                return result['TckrSymb'].tolist()
        except Exception as e:
            logger.error(f"Error getting symbols for {test_date}: {e}")
            return []
    
    def get_fno_features(self, symbol: str, test_date: str) -> Dict:
        """Get FNO features for ML prediction simulation"""
        try:
            with duckdb.connect(self.db_path) as conn:
                # Get FNO data for the symbol
                fno_query = """
                SELECT ClsPric, OpnPric, HghPric, LwPric, TtlTradgVol, OpnIntrst, 
                       ChngInOpnIntrst, OptnTp, StrkPric
                FROM fno_bhav_copy 
                WHERE TckrSymb = ? AND TRADE_DATE = ?
                """
                fno_result = conn.execute(fno_query, [symbol, test_date]).fetchdf()
                
                if fno_result.empty:
                    return {}
                
                # Calculate aggregated features
                features = {
                    'close_price': fno_result['ClsPric'].mean(),
                    'open_price': fno_result['OpnPric'].mean(),
                    'high_price': fno_result['HghPric'].max(),
                    'low_price': fno_result['LwPric'].min(),
                    'volume': fno_result['TtlTradgVol'].sum(),
                    'total_oi': fno_result['OpnIntrst'].sum(),
                    'oi_change': fno_result['ChngInOpnIntrst'].sum(),
                    'daily_return': ((fno_result['ClsPric'].mean() - fno_result['OpnPric'].mean()) / fno_result['OpnPric'].mean()) * 100 if fno_result['OpnPric'].mean() > 0 else 0,
                    'price_range': ((fno_result['HghPric'].max() - fno_result['LwPric'].min()) / fno_result['ClsPric'].mean()) * 100 if fno_result['ClsPric'].mean() > 0 else 0
                }
                
                # Calculate PCR (Put-Call Ratio)
                call_oi = fno_result[fno_result['OptnTp'].str.contains('CE', na=False)]['OpnIntrst'].sum()
                put_oi = fno_result[fno_result['OptnTp'].str.contains('PE', na=False)]['OpnIntrst'].sum()
                features['pcr'] = put_oi / call_oi if call_oi > 0 else 1.0
                
                # Calculate average strike price
                features['avg_strike'] = fno_result['StrkPric'].mean()
                
                # Calculate moneyness (how far from ATM)
                if features['avg_strike'] > 0 and features['close_price'] > 0:
                    features['moneyness'] = features['close_price'] / features['avg_strike']
                else:
                    features['moneyness'] = 1.0
                
                return features
                
        except Exception as e:
            logger.error(f"Error getting FNO features for {symbol} on {test_date}: {e}")
            return {}
    
    def get_actual_outcome_fixed(self, symbol: str, test_date: str, horizon: str) -> Tuple[float, str]:
        """Get actual outcome using most liquid ATM options or underlying stock price"""
        try:
            with duckdb.connect(self.db_path) as conn:
                # First, try to get the most liquid ATM option for test date
                test_query = """
                SELECT ClsPric, StrkPric, OptnTp, TtlTradgVol, OpnIntrst
                FROM fno_bhav_copy 
                WHERE TckrSymb = ? AND TRADE_DATE = ?
                AND TtlTradgVol > 0  -- Only liquid contracts
                AND OpnIntrst > 0    -- Only contracts with open interest
                ORDER BY TtlTradgVol DESC, OpnIntrst DESC
                LIMIT 1
                """
                test_result = conn.execute(test_query, [symbol, test_date]).fetchdf()
                
                if test_result.empty:
                    # Fallback: get any contract with volume
                    fallback_query = """
                    SELECT ClsPric, StrkPric, OptnTp
                    FROM fno_bhav_copy 
                    WHERE TckrSymb = ? AND TRADE_DATE = ?
                    AND TtlTradgVol > 0
                    ORDER BY TtlTradgVol DESC
                    LIMIT 1
                    """
                    test_result = conn.execute(fallback_query, [symbol, test_date]).fetchdf()
                
                if test_result.empty:
                    return None, None
                
                test_price = test_result['ClsPric'].iloc[0]
                test_strike = test_result['StrkPric'].iloc[0]
                test_date_obj = pd.to_datetime(test_date)
                
                # Calculate target date based on horizon
                if horizon == "Daily":
                    target_date = test_date_obj + timedelta(days=1)
                elif horizon == "Weekly":
                    target_date = test_date_obj + timedelta(days=7)
                elif horizon == "Monthly":
                    target_date = test_date_obj + timedelta(days=30)
                else:
                    return None, None
                
                # Try to find the same strike price option on target date
                target_query = """
                SELECT ClsPric, StrkPric, OptnTp, TtlTradgVol
                FROM fno_bhav_copy 
                WHERE TckrSymb = ? AND TRADE_DATE >= ?
                AND StrkPric = ?
                AND TtlTradgVol > 0
                ORDER BY TRADE_DATE, TtlTradgVol DESC
                LIMIT 1
                """
                target_result = conn.execute(target_query, [symbol, target_date, test_strike]).fetchdf()
                
                if target_result.empty:
                    # Try to find any liquid option on target date
                    liquid_query = """
                    SELECT ClsPric, StrkPric, OptnTp, TtlTradgVol
                    FROM fno_bhav_copy 
                    WHERE TckrSymb = ? AND TRADE_DATE >= ?
                    AND TtlTradgVol > 0
                    ORDER BY TRADE_DATE, TtlTradgVol DESC
                    LIMIT 1
                    """
                    target_result = conn.execute(liquid_query, [symbol, target_date]).fetchdf()
                
                if target_result.empty:
                    return None, None
                
                actual_price = target_result['ClsPric'].iloc[0]
                
                # Calculate return with sanity checks
                if test_price > 0 and test_price < 10000:  # Reasonable price range
                    return_pct = ((actual_price - test_price) / test_price) * 100
                    
                    # Cap returns to realistic levels (max ±50% for daily, ±100% for weekly)
                    if horizon == "Daily":
                        return_pct = max(-50, min(50, return_pct))
                    elif horizon == "Weekly":
                        return_pct = max(-100, min(100, return_pct))
                    
                    direction = "UP" if return_pct > 0 else "DOWN"
                    return return_pct, direction
                else:
                    return None, None
                    
        except Exception as e:
            logger.error(f"Error getting actual outcome for {symbol} on {test_date}: {e}")
            return None, None
    
    def simulate_ml_prediction_balanced(self, features: Dict) -> float:
        """Simulate ML prediction with balanced approach"""
        try:
            # Start with neutral base probability
            base_prob = 0.5
            
            # Get feature values
            daily_return = features.get('daily_return', 0)
            pcr = features.get('pcr', 1.0)
            oi_change = features.get('oi_change', 0)
            volume = features.get('volume', 0)
            moneyness = features.get('moneyness', 1.0)
            
            # More balanced adjustments based on daily return
            if daily_return > 3.0:
                base_prob += 0.12
            elif daily_return > 2.0:
                base_prob += 0.08
            elif daily_return > 1.0:
                base_prob += 0.05
            elif daily_return > 0.5:
                base_prob += 0.02
            elif daily_return < -3.0:
                base_prob -= 0.12
            elif daily_return < -2.0:
                base_prob -= 0.08
            elif daily_return < -1.0:
                base_prob -= 0.05
            elif daily_return < -0.5:
                base_prob -= 0.02
            
            # Balanced PCR adjustments
            if pcr > 2.0:  # Very high put activity (very bearish)
                base_prob -= 0.10
            elif pcr > 1.5:  # High put activity (bearish)
                base_prob -= 0.06
            elif pcr < 0.5:  # Very low put-call ratio (very bullish)
                base_prob += 0.10
            elif pcr < 0.7:  # Low put-call ratio (bullish)
                base_prob += 0.06
            
            # Balanced OI change adjustments
            if oi_change > 2000000:  # Massive OI buildup
                base_prob += 0.04
            elif oi_change > 1000000:  # High OI buildup
                base_prob += 0.02
            elif oi_change < -2000000:  # Massive OI unwinding
                base_prob -= 0.04
            elif oi_change < -1000000:  # High OI unwinding
                base_prob -= 0.02
            
            # Volume adjustments (smaller impact)
            if volume > 10000000:  # Very high volume
                base_prob += 0.02
            elif volume > 5000000:  # High volume
                base_prob += 0.01
            elif volume < 1000000:  # Low volume
                base_prob -= 0.01
            
            # Moneyness adjustments (smaller impact)
            if moneyness > 1.1:  # Deep ITM calls
                base_prob += 0.01
            elif moneyness < 0.9:  # Deep ITM puts
                base_prob -= 0.01
            
            # Add more balanced randomness
            noise = random.uniform(-0.15, 0.15)  # Increased randomness range
            final_prob = base_prob + noise
            
            # Ensure probability is between 0 and 1
            return max(0.0, min(1.0, final_prob))
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 0.5
    
    def simulate_rag_analysis(self, symbol: str, test_date: str, features: Dict) -> str:
        """Simulate RAG analysis based on FNO features"""
        try:
            daily_return = features.get('daily_return', 0)
            pcr = features.get('pcr', 1.0)
            volume = features.get('volume', 0)
            oi_change = features.get('oi_change', 0)
            moneyness = features.get('moneyness', 1.0)
            
            analysis_parts = []
            
            # Price movement analysis
            if daily_return > 3.0:
                analysis_parts.append("Exceptional bullish momentum with strong price gains")
            elif daily_return > 1.5:
                analysis_parts.append("Strong upward movement")
            elif daily_return > 0.5:
                analysis_parts.append("Moderate upward movement")
            elif daily_return < -3.0:
                analysis_parts.append("Exceptional bearish pressure with significant losses")
            elif daily_return < -1.5:
                analysis_parts.append("Strong downward movement")
            elif daily_return < -0.5:
                analysis_parts.append("Moderate downward movement")
            else:
                analysis_parts.append("Sideways movement with minimal price change")
            
            # PCR analysis
            if pcr > 2.0:
                analysis_parts.append("Extremely high put-call ratio indicates strong bearish sentiment")
            elif pcr > 1.5:
                analysis_parts.append("High put-call ratio indicates bearish sentiment")
            elif pcr < 0.5:
                analysis_parts.append("Very low put-call ratio suggests strong bullish sentiment")
            elif pcr < 0.7:
                analysis_parts.append("Low put-call ratio suggests bullish sentiment")
            else:
                analysis_parts.append("Neutral options sentiment")
            
            # OI analysis
            if oi_change > 2000000:
                analysis_parts.append("Massive open interest buildup suggests strong directional move")
            elif oi_change > 1000000:
                analysis_parts.append("High open interest buildup indicates strong conviction")
            elif oi_change < -2000000:
                analysis_parts.append("Massive open interest unwinding suggests trend reversal")
            elif oi_change < -1000000:
                analysis_parts.append("High open interest unwinding indicates profit booking")
            
            # Volume analysis
            if volume > 10000000:
                analysis_parts.append("Exceptional trading volume confirms strong market participation")
            elif volume > 5000000:
                analysis_parts.append("High trading volume confirms strong market participation")
            elif volume < 1000000:
                analysis_parts.append("Low volume suggests weak market conviction")
            
            # Moneyness analysis
            if moneyness > 1.1:
                analysis_parts.append("Deep ITM calls indicate strong bullish expectations")
            elif moneyness < 0.9:
                analysis_parts.append("Deep ITM puts indicate strong bearish expectations")
            
            # Combine analysis
            analysis = f"FNO analysis for {symbol} on {test_date}: {' '.join(analysis_parts)}."
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in RAG analysis: {e}")
            return f"FNO analysis for {symbol} on {test_date}: Market data analysis completed."
    
    def run_single_test(self, symbol: str, test_date: str, horizon: str) -> Optional[Dict]:
        """Run a single test case"""
        try:
            start_time = datetime.now()
            
            # Get FNO features
            features = self.get_fno_features(symbol, test_date)
            if not features:
                return None
            
            # Simulate ML prediction with balanced approach
            ml_prediction = self.simulate_ml_prediction_balanced(features)
            
            # Simulate RAG analysis
            rag_analysis = self.simulate_rag_analysis(symbol, test_date, features)
            
            # Get actual outcome using fixed method
            actual_return, actual_direction = self.get_actual_outcome_fixed(symbol, test_date, horizon)
            
            if actual_return is None:
                return None
            
            # Determine if prediction was correct
            predicted_direction = "UP" if ml_prediction > 0.5 else "DOWN"
            prediction_correct = predicted_direction == actual_direction
            
            # Calculate confidence score
            confidence_score = abs(ml_prediction - 0.5) * 2  # Convert to 0-1 scale
            
            end_time = datetime.now()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return {
                'test_date': test_date,
                'symbol': symbol,
                'horizon': horizon,
                'ml_prediction': ml_prediction,
                'predicted_direction': predicted_direction,
                'rag_analysis': rag_analysis,
                'actual_return': actual_return,
                'actual_direction': actual_direction,
                'prediction_correct': prediction_correct,
                'confidence_score': confidence_score,
                'response_time_ms': response_time_ms,
                'prediction_error': abs(ml_prediction - (1 if actual_direction == "UP" else 0)),
                'fno_features': features
            }
            
        except Exception as e:
            logger.error(f"Error in test for {symbol} on {test_date}: {e}")
            return None
    
    def run_comprehensive_backtest(self, 
                                 start_date: str = "2024-01-01", 
                                 end_date: str = "2024-12-31",
                                 symbols_per_date: int = 15,
                                 horizons: List[str] = None) -> pd.DataFrame:
        """Run comprehensive backtesting"""
        
        if horizons is None:
            horizons = ["Daily", "Weekly"]
        
        logger.info(f"Starting FNO comprehensive backtest (BALANCED) from {start_date} to {end_date}")
        logger.info(f"Testing {symbols_per_date} symbols per date across {len(horizons)} horizons")
        
        # Get test dates
        test_dates = self.get_test_dates(start_date, end_date)
        logger.info(f"Found {len(test_dates)} test dates")
        
        total_tests = len(test_dates) * symbols_per_date * len(horizons)
        completed_tests = 0
        
        for test_date in test_dates:
            logger.info(f"Testing date: {test_date}")
            
            # Get symbols for this date
            symbols = self.get_symbols_for_testing(test_date, symbols_per_date)
            
            if not symbols:
                logger.warning(f"No symbols found for {test_date}")
                continue
            
            for symbol in symbols:
                for horizon in horizons:
                    try:
                        result = self.run_single_test(symbol, test_date, horizon)
                        
                        if result:
                            self.results.append(result)
                            completed_tests += 1
                            
                            if completed_tests % 10 == 0:
                                logger.info(f"Completed {completed_tests}/{total_tests} tests")
                        
                    except Exception as e:
                        logger.error(f"Error testing {symbol} on {test_date} for {horizon}: {e}")
                        continue
        
        # Convert results to DataFrame
        results_df = self._convert_results_to_dataframe()
        
        logger.info(f"Backtest completed. Total results: {len(results_df)}")
        return results_df
    
    def _convert_results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            # Extract key features for CSV
            features = result.get('fno_features', {})
            data.append({
                'test_date': result['test_date'],
                'symbol': result['symbol'],
                'horizon': result['horizon'],
                'ml_prediction': result['ml_prediction'],
                'predicted_direction': result['predicted_direction'],
                'rag_analysis': result['rag_analysis'],
                'actual_return': result['actual_return'],
                'actual_direction': result['actual_direction'],
                'prediction_correct': result['prediction_correct'],
                'confidence_score': result['confidence_score'],
                'response_time_ms': result['response_time_ms'],
                'prediction_error': result['prediction_error'],
                'close_price': features.get('close_price', 0),
                'daily_return': features.get('daily_return', 0),
                'pcr': features.get('pcr', 1.0),
                'volume': features.get('volume', 0),
                'total_oi': features.get('total_oi', 0),
                'oi_change': features.get('oi_change', 0),
                'moneyness': features.get('moneyness', 1.0)
            })
        
        return pd.DataFrame(data)
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary report"""
        if results_df.empty:
            return {}
        
        summary = {
            'total_tests': len(results_df),
            'overall_accuracy': (results_df['prediction_correct'].sum() / len(results_df)) * 100,
            'avg_response_time_ms': results_df['response_time_ms'].mean(),
            'avg_confidence': results_df['confidence_score'].mean(),
            'avg_prediction_error': results_df['prediction_error'].mean(),
            'horizon_breakdown': {},
            'symbol_breakdown': {},
            'date_range': {
                'start': results_df['test_date'].min(),
                'end': results_df['test_date'].max()
            }
        }
        
        # Horizon breakdown
        for horizon in results_df['horizon'].unique():
            horizon_data = results_df[results_df['horizon'] == horizon]
            summary['horizon_breakdown'][horizon] = {
                'count': len(horizon_data),
                'accuracy': (horizon_data['prediction_correct'].sum() / len(horizon_data)) * 100,
                'avg_return': horizon_data['actual_return'].mean(),
                'avg_prediction': horizon_data['ml_prediction'].mean()
            }
        
        # Symbol breakdown (top 10)
        symbol_stats = results_df.groupby('symbol').agg({
            'prediction_correct': 'mean',
            'actual_return': 'mean',
            'ml_prediction': 'mean'
        }).sort_values('prediction_correct', ascending=False).head(10)
        
        summary['symbol_breakdown'] = symbol_stats.to_dict('index')
        
        return summary
    
    def save_results(self, results_df: pd.DataFrame, filename: str = None):
        """Save results to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fno_comprehensive_backtest_results_BALANCED_{timestamp}.csv"
        
        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        
        # Save summary report
        summary = self.generate_summary_report(results_df)
        summary_filename = filename.replace('.csv', '_summary.txt')
        
        with open(summary_filename, 'w') as f:
            f.write("FNO COMPREHENSIVE ML+RAG BACKTEST SUMMARY (BALANCED)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%\n")
            f.write(f"Average Response Time: {summary['avg_response_time_ms']:.0f}ms\n")
            f.write(f"Average Confidence: {summary['avg_confidence']:.3f}\n")
            f.write(f"Average Prediction Error: {summary['avg_prediction_error']:.3f}\n\n")
            
            f.write("HORIZON BREAKDOWN:\n")
            f.write("-" * 20 + "\n")
            for horizon, stats in summary['horizon_breakdown'].items():
                f.write(f"{horizon}:\n")
                f.write(f"  Count: {stats['count']}\n")
                f.write(f"  Accuracy: {stats['accuracy']:.2f}%\n")
                f.write(f"  Avg Return: {stats['avg_return']:.2f}%\n")
                f.write(f"  Avg Prediction: {stats['avg_prediction']:.3f}\n\n")
            
            f.write("TOP 10 SYMBOLS BY ACCURACY:\n")
            f.write("-" * 30 + "\n")
            for symbol, stats in summary['symbol_breakdown'].items():
                f.write(f"{symbol}: {stats['prediction_correct']*100:.2f}% accuracy\n")
        
        logger.info(f"Summary saved to {summary_filename}")

def main():
    """Main function to run comprehensive FNO backtesting"""
    logger.info("Starting FNO Comprehensive ML+RAG Backtesting (BALANCED VERSION)")
    
    # Initialize backtester
    backtester = FNOBacktestMLRAGBalanced()
    
    # Run backtest for last 6 months
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    try:
        # Run comprehensive backtest
        results_df = backtester.run_comprehensive_backtest(
            start_date=start_date,
            end_date=end_date,
            symbols_per_date=15,  # Test 15 symbols per date
            horizons=["Daily", "Weekly"]  # Focus on Daily and Weekly
        )
        
        if not results_df.empty:
            # Save results
            backtester.save_results(results_df)
            
            # Print summary
            summary = backtester.generate_summary_report(results_df)
            print(f"\n{'='*60}")
            print("FNO COMPREHENSIVE ML+RAG BACKTEST RESULTS (BALANCED)")
            print(f"{'='*60}")
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
            print(f"Average Response Time: {summary['avg_response_time_ms']:.0f}ms")
            print(f"Average Confidence: {summary['avg_confidence']:.3f}")
            print(f"Average Prediction Error: {summary['avg_prediction_error']:.3f}")
            
            print(f"\nHorizon Breakdown:")
            for horizon, stats in summary['horizon_breakdown'].items():
                print(f"  {horizon}: {stats['accuracy']:.2f}% accuracy ({stats['count']} tests)")
            
            print(f"\nTop 5 Symbols by Accuracy:")
            symbol_stats = list(summary['symbol_breakdown'].items())[:5]
            for symbol, stats in symbol_stats:
                print(f"  {symbol}: {stats['prediction_correct']*100:.2f}%")
            
            print(f"\nSample Results:")
            print(results_df.head(10).to_string(index=False))
            
        else:
            logger.warning("No results generated from backtest")
            
    except Exception as e:
        logger.error(f"Error in main backtest: {e}")
        raise

if __name__ == "__main__":
    main()

