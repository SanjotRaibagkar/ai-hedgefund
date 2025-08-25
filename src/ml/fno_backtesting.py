"""
FNO ML Strategy Backtesting Module.
Comprehensive backtesting for FNO-only ML strategy with multi-horizon predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import duckdb
from src.ml.fno_ml_strategy import FNOMLStrategy
from src.tools.enhanced_api import get_fno_bhav_copy_data


class FNOMLBacktesting:
    """Comprehensive backtesting framework for FNO ML strategy."""
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
        """
        Initialize FNO ML backtesting framework.
        
        Args:
            initial_capital: Initial portfolio capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Backtesting components
        self.fno_strategy = FNOMLStrategy()
        
        # Backtesting state
        self.portfolio_state = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        # Results storage
        self.backtest_results = {}
        self.prediction_vs_actual = {}
        
        logger.info("FNOMLBacktesting initialized")
    
    def get_top_20_tickers(self) -> List[str]:
        """Get top 20 tickers with good FNO data."""
        try:
            conn = duckdb.connect('data/comprehensive_equity.duckdb')
            
            # Get top 20 tickers with most FNO data
            tickers = conn.execute('''
                SELECT TckrSymb, COUNT(*) as fno_count, 
                       COUNT(DISTINCT TRADE_DATE) as trading_days,
                       MIN(TRADE_DATE) as start_date, 
                       MAX(TRADE_DATE) as end_date
                FROM fno_bhav_copy 
                GROUP BY TckrSymb 
                HAVING COUNT(*) > 10000 AND COUNT(DISTINCT TRADE_DATE) > 200
                ORDER BY COUNT(*) DESC 
                LIMIT 20
            ''').fetchall()
            
            conn.close()
            
            ticker_list = [row[0] for row in tickers]
            logger.info(f"Selected {len(ticker_list)} tickers for backtesting: {ticker_list}")
            
            return ticker_list
            
        except Exception as e:
            logger.error(f"Error getting top tickers: {e}")
            # Fallback to known good tickers
            return ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTYNXT50", 
                   "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                   "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "AXISBANK",
                   "ASIANPAINT", "MARUTI", "SUNPHARMA", "TATAMOTORS", "WIPRO"]
    
    def run_comprehensive_backtest(self,
                                  start_date: str = "2025-07-01",
                                  end_date: Optional[str] = None,
                                  tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest for multiple tickers.
        
        Args:
            start_date: Start date for predictions (default: July 1, 2025)
            end_date: End date for backtest (default: today)
            tickers: List of tickers to test (default: top 20)
            
        Returns:
            Comprehensive backtest results
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if tickers is None:
                tickers = self.get_top_20_tickers()
            
            logger.info(f"Starting comprehensive backtest for {len(tickers)} tickers")
            logger.info(f"Period: {start_date} to {end_date}")
            
            # Initialize results storage
            self.backtest_results = {}
            self.prediction_vs_actual = {}
            
            # Training period (6 months before prediction start)
            train_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')
            
            for ticker in tickers:
                try:
                    logger.info(f"Processing ticker: {ticker}")
                    
                    # Run backtest for this ticker
                    ticker_result = self._run_ticker_backtest(
                        ticker=ticker,
                        train_start=train_start,
                        train_end=start_date,
                        predict_start=start_date,
                        predict_end=end_date
                    )
                    
                    self.backtest_results[ticker] = ticker_result
                    
                except Exception as e:
                    logger.error(f"Backtest failed for {ticker}: {e}")
                    self.backtest_results[ticker] = {'error': str(e)}
            
            # Generate comprehensive report
            report = self._generate_comprehensive_report()
            
            # Save results to CSV
            csv_path = self._save_results_to_csv()
            
            logger.info("Comprehensive backtest completed")
            
            return {
                'backtest_period': {'start': start_date, 'end': end_date},
                'tickers_tested': len(tickers),
                'backtest_results': self.backtest_results,
                'prediction_vs_actual': self.prediction_vs_actual,
                'comprehensive_report': report,
                'csv_file_path': csv_path
            }
            
        except Exception as e:
            logger.error(f"Comprehensive backtest failed: {e}")
            return {'error': str(e)}
    
    def _run_ticker_backtest(self,
                            ticker: str,
                            train_start: str,
                            train_end: str,
                            predict_start: str,
                            predict_end: str) -> Dict[str, Any]:
        """Run backtest for a single ticker."""
        try:
            logger.info(f"Running backtest for {ticker}")
            
            # Initialize strategy
            strategy_config = {
                'min_data_points': 50,
                'min_oi_threshold': 100,
                'min_volume_threshold': 50,
                'confidence_threshold': 0.7,
                'price_change_threshold': 0.02
            }
            
            strategy = FNOMLStrategy(strategy_config=strategy_config)
            
            # Train models
            logger.info(f"Training models for {ticker}")
            training_results = strategy.train_models(
                ticker=ticker,
                start_date=train_start,
                end_date=train_end
            )
            
            if 'error' in training_results:
                raise ValueError(f"Training failed: {training_results['error']}")
            
            # Get predictions
            logger.info(f"Getting predictions for {ticker}")
            predictions = strategy.predict_returns(
                ticker=ticker,
                start_date=predict_start,
                end_date=predict_end
            )
            
            if 'error' in predictions:
                raise ValueError(f"Prediction failed: {predictions['error']}")
            
            # Get actual data for comparison
            actual_data = self._get_actual_data(ticker, predict_start, predict_end)
            
            if actual_data.empty:
                raise ValueError(f"No actual data available for {ticker}")
            
            # Compare predictions with actual results
            comparison_results = self._compare_predictions_vs_actual(
                ticker, predictions, actual_data
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_ticker_performance(
                ticker, comparison_results
            )
            
            # Store prediction vs actual data
            self.prediction_vs_actual[ticker] = comparison_results
            
            logger.info(f"Backtest completed for {ticker}")
            
            return {
                'ticker': ticker,
                'training_results': training_results,
                'predictions': predictions,
                'actual_data': actual_data,
                'comparison_results': comparison_results,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Ticker backtest failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _get_actual_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get actual price data for comparison."""
        try:
            # Get FNO data and calculate daily average prices
            fno_data = get_fno_bhav_copy_data(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if fno_data.empty:
                return pd.DataFrame()
            
            # Calculate daily average closing prices
            daily_prices = fno_data.groupby('TRADE_DATE')['ClsPric'].mean().reset_index()
            daily_prices['TRADE_DATE'] = pd.to_datetime(daily_prices['TRADE_DATE'])
            daily_prices = daily_prices.sort_values('TRADE_DATE')
            
            # Calculate actual returns
            daily_prices['actual_return_1d'] = daily_prices['ClsPric'].pct_change(1)
            daily_prices['actual_return_5d'] = daily_prices['ClsPric'].pct_change(5)
            daily_prices['actual_return_21d'] = daily_prices['ClsPric'].pct_change(21)
            
            return daily_prices
            
        except Exception as e:
            logger.error(f"Error getting actual data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _compare_predictions_vs_actual(self,
                                      ticker: str,
                                      predictions: Dict[str, Any],
                                      actual_data: pd.DataFrame) -> pd.DataFrame:
        """Compare predictions with actual results."""
        try:
            comparison_data = []
            
            for horizon in ['1d', '5d', '21d']:
                if horizon in predictions and 'predictions' in predictions[horizon]:
                    pred_df = predictions[horizon]['predictions']
                    
                    # Merge predictions with actual data
                    merged_data = pd.merge(
                        pred_df,
                        actual_data[['TRADE_DATE', f'actual_return_{horizon}']],
                        left_on='date',
                        right_on='TRADE_DATE',
                        how='inner'
                    )
                    
                    # Calculate comparison metrics
                    merged_data['prediction_error'] = (
                        merged_data['predicted_return'] - merged_data[f'actual_return_{horizon}']
                    )
                    merged_data['absolute_error'] = abs(merged_data['prediction_error'])
                    merged_data['squared_error'] = merged_data['prediction_error'] ** 2
                    
                    # Add horizon information
                    merged_data['horizon'] = horizon
                    merged_data['ticker'] = ticker
                    
                    comparison_data.append(merged_data)
            
            if comparison_data:
                return pd.concat(comparison_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error comparing predictions for {ticker}: {e}")
            return pd.DataFrame()
    
    def _calculate_ticker_performance(self,
                                     ticker: str,
                                     comparison_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for a ticker."""
        try:
            if comparison_data.empty:
                return {'error': 'No comparison data available'}
            
            performance_metrics = {}
            
            for horizon in ['1d', '5d', '21d']:
                horizon_data = comparison_data[comparison_data['horizon'] == horizon]
                
                if not horizon_data.empty:
                    # Calculate accuracy metrics
                    mae = horizon_data['absolute_error'].mean()
                    mse = horizon_data['squared_error'].mean()
                    rmse = np.sqrt(mse)
                    
                    # Calculate directional accuracy
                    correct_direction = (
                        (horizon_data['predicted_return'] > 0) & 
                        (horizon_data[f'actual_return_{horizon}'] > 0)
                    ) | (
                        (horizon_data['predicted_return'] < 0) & 
                        (horizon_data[f'actual_return_{horizon}'] < 0)
                    )
                    directional_accuracy = correct_direction.mean()
                    
                    # Calculate correlation
                    correlation = horizon_data['predicted_return'].corr(
                        horizon_data[f'actual_return_{horizon}']
                    )
                    
                    performance_metrics[horizon] = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'directional_accuracy': directional_accuracy,
                        'correlation': correlation,
                        'total_predictions': len(horizon_data)
                    }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance for {ticker}: {e}")
            return {'error': str(e)}
    
    def _generate_comprehensive_report(self) -> str:
        """Generate comprehensive backtest report."""
        try:
            successful_tickers = [
                ticker for ticker, result in self.backtest_results.items()
                if 'error' not in result
            ]
            
            failed_tickers = [
                ticker for ticker, result in self.backtest_results.items()
                if 'error' in result
            ]
            
            report = f"""
FNO ML Strategy Comprehensive Backtest Report
============================================

Backtest Summary:
----------------
Total Tickers Tested: {len(self.backtest_results)}
Successful Backtests: {len(successful_tickers)}
Failed Backtests: {len(failed_tickers)}
Success Rate: {len(successful_tickers)/len(self.backtest_results)*100:.1f}%

Successful Tickers: {', '.join(successful_tickers)}
Failed Tickers: {', '.join(failed_tickers)}

Performance Summary by Horizon:
-----------------------------
"""
            
            # Aggregate performance across all tickers
            horizon_performance = {'1d': [], '5d': [], '21d': []}
            
            for ticker, result in self.backtest_results.items():
                if 'error' not in result and 'performance_metrics' in result:
                    perf_metrics = result['performance_metrics']
                    for horizon in ['1d', '5d', '21d']:
                        if horizon in perf_metrics:
                            horizon_performance[horizon].append(perf_metrics[horizon])
            
            for horizon in ['1d', '5d', '21d']:
                if horizon_performance[horizon]:
                    avg_mae = np.mean([p['mae'] for p in horizon_performance[horizon]])
                    avg_rmse = np.mean([p['rmse'] for p in horizon_performance[horizon]])
                    avg_directional_accuracy = np.mean([p['directional_accuracy'] for p in horizon_performance[horizon]])
                    avg_correlation = np.mean([p['correlation'] for p in horizon_performance[horizon]])
                    
                    report += f"""
{horizon} Horizon:
- Average MAE: {avg_mae:.4f}
- Average RMSE: {avg_rmse:.4f}
- Average Directional Accuracy: {avg_directional_accuracy:.2%}
- Average Correlation: {avg_correlation:.4f}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Report generation failed: {e}"
    
    def _save_results_to_csv(self) -> str:
        """Save backtest results to CSV file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"fno_ml_backtest_results_{timestamp}.csv"
            csv_path = f"data/backtest_results/{csv_filename}"
            
            # Create directory if it doesn't exist
            import os
            os.makedirs("data/backtest_results", exist_ok=True)
            
            # Prepare data for CSV
            csv_data = []
            
            for ticker, result in self.backtest_results.items():
                if 'error' not in result and 'comparison_results' in result:
                    comparison_df = result['comparison_results']
                    
                    for _, row in comparison_df.iterrows():
                        csv_data.append({
                            'ticker': ticker,
                            'date': row['date'],
                            'horizon': row['horizon'],
                            'predicted_return': row['predicted_return'],
                            'actual_return': row[f'actual_return_{row["horizon"]}'],
                            'prediction_error': row['prediction_error'],
                            'absolute_error': row['absolute_error'],
                            'confidence': row.get('confidence', np.nan),
                            'prediction_direction': 'positive' if row['predicted_return'] > 0 else 'negative',
                            'actual_direction': 'positive' if row[f'actual_return_{row["horizon"]}'] > 0 else 'negative',
                            'direction_correct': (
                                (row['predicted_return'] > 0 and row[f'actual_return_{row["horizon"]}'] > 0) or
                                (row['predicted_return'] < 0 and row[f'actual_return_{row["horizon"]}'] < 0)
                            )
                        })
            
            # Create DataFrame and save to CSV
            if csv_data:
                results_df = pd.DataFrame(csv_data)
                results_df.to_csv(csv_path, index=False)
                logger.info(f"Backtest results saved to: {csv_path}")
                return csv_path
            else:
                logger.warning("No data to save to CSV")
                return ""
                
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            return ""
    
    def get_detailed_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific ticker."""
        try:
            if ticker not in self.backtest_results:
                return {'error': f'No backtest results for {ticker}'}
            
            result = self.backtest_results[ticker]
            
            if 'error' in result:
                return result
            
            # Get comparison data
            comparison_df = result.get('comparison_results', pd.DataFrame())
            
            if comparison_df.empty:
                return {'error': f'No comparison data for {ticker}'}
            
            # Calculate detailed metrics
            detailed_analysis = {
                'ticker': ticker,
                'performance_metrics': result.get('performance_metrics', {}),
                'horizon_analysis': {}
            }
            
            for horizon in ['1d', '5d', '21d']:
                horizon_data = comparison_df[comparison_df['horizon'] == horizon]
                
                if not horizon_data.empty:
                    detailed_analysis['horizon_analysis'][horizon] = {
                        'total_predictions': len(horizon_data),
                        'mae': horizon_data['absolute_error'].mean(),
                        'rmse': np.sqrt(horizon_data['squared_error'].mean()),
                        'directional_accuracy': horizon_data['direction_correct'].mean(),
                        'correlation': horizon_data['predicted_return'].corr(
                            horizon_data[f'actual_return_{horizon}']
                        ),
                        'best_prediction': horizon_data.loc[
                            horizon_data['absolute_error'].idxmin()
                        ].to_dict(),
                        'worst_prediction': horizon_data.loc[
                            horizon_data['absolute_error'].idxmax()
                        ].to_dict()
                    }
            
            return detailed_analysis
            
        except Exception as e:
            logger.error(f"Error getting detailed analysis for {ticker}: {e}")
            return {'error': str(e)}


def run_fno_backtest():
    """Run the FNO ML strategy backtest."""
    print("🚀 FNO ML Strategy Comprehensive Backtest")
    print("=" * 80)
    
    try:
        # Initialize backtesting framework
        backtester = FNOMLBacktesting()
        
        # Run comprehensive backtest
        results = backtester.run_comprehensive_backtest(
            start_date="2025-07-01",
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if 'error' in results:
            print(f"❌ Backtest failed: {results['error']}")
            return
        
        # Print summary
        print(f"\n✅ Backtest completed successfully!")
        print(f"📊 Tickers tested: {results['tickers_tested']}")
        print(f"📅 Period: {results['backtest_period']['start']} to {results['backtest_period']['end']}")
        print(f"📁 Results saved to: {results['csv_file_path']}")
        
        # Print comprehensive report
        print(f"\n📋 COMPREHENSIVE REPORT:")
        print(results['comprehensive_report'])
        
        # Print top performers
        print(f"\n🏆 TOP PERFORMERS BY DIRECTIONAL ACCURACY:")
        ticker_performance = []
        
        for ticker, result in results['backtest_results'].items():
            if 'error' not in result and 'performance_metrics' in result:
                perf_metrics = result['performance_metrics']
                avg_directional_accuracy = 0
                count = 0
                
                for horizon in ['1d', '5d', '21d']:
                    if horizon in perf_metrics:
                        avg_directional_accuracy += perf_metrics[horizon]['directional_accuracy']
                        count += 1
                
                if count > 0:
                    avg_directional_accuracy /= count
                    ticker_performance.append((ticker, avg_directional_accuracy))
        
        # Sort by performance
        ticker_performance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (ticker, accuracy) in enumerate(ticker_performance[:10], 1):
            print(f"   {i:2d}. {ticker}: {accuracy:.2%}")
        
        print(f"\n✅ FNO ML Strategy Backtest completed successfully!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_fno_backtest()
