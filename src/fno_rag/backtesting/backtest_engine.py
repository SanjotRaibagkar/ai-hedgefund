#!/usr/bin/env python3
"""
FNO RAG Backtesting Engine
Comprehensive backtesting framework for FNO RAG system validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from loguru import logger
from dataclasses import dataclass, asdict
from pathlib import Path

from ..core.fno_engine import FNOEngine
from ..models.data_models import HorizonType, PredictionRequest, ProbabilityResult
from ...data.database.duckdb_manager import DatabaseManager


@dataclass
class BacktestResult:
    """Individual backtest result."""
    symbol: str
    prediction_date: str
    horizon: str
    predicted_up_prob: float
    predicted_down_prob: float
    predicted_neutral_prob: float
    confidence_score: float
    actual_return: float
    actual_direction: str  # 'up', 'down', 'neutral'
    prediction_correct: bool
    prediction_accuracy: float  # How close the prediction was to actual
    days_to_target: int  # Days from prediction to actual outcome


@dataclass
class BacktestSummary:
    """Summary of backtest results."""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    avg_prediction_accuracy: float
    symbol_accuracy: Dict[str, float]
    horizon_accuracy: Dict[str, float]
    date_range: Tuple[str, str]
    test_dates: List[str]


class FNOBacktestEngine:
    """Backtesting engine for FNO RAG system."""
    
    def __init__(self, fno_engine: Optional[FNOEngine] = None):
        """Initialize backtesting engine."""
        self.fno_engine = fno_engine or FNOEngine()
        self.db_manager = DatabaseManager()
        self.logger = logger
        
        # Create results directory
        self.results_dir = Path("data/backtest_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical FNO data for backtesting."""
        try:
            query = """
                SELECT 
                    TckrSymb as symbol,
                    TRADE_DATE as date,
                    OpnPric as open_price,
                    HghPric as high_price,
                    LwPric as low_price,
                    ClsPric as close_price,
                    TtlTradgVol as volume,
                    OpnIntrst as open_interest
                FROM fno_bhav_copy
                WHERE TckrSymb IN ({})
                AND TRADE_DATE BETWEEN '{}' AND '{}'
                ORDER BY TckrSymb, TRADE_DATE
            """.format(
                ','.join([f"'{s}'" for s in symbols]),
                start_date,
                end_date
            )
            
            df = self.db_manager.connection.execute(query).fetchdf()
            self.logger.info(f"Retrieved {len(df)} historical records for {len(symbols)} symbols")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def calculate_actual_returns(self, df: pd.DataFrame, horizon: HorizonType) -> pd.DataFrame:
        """Calculate actual returns for different horizons."""
        try:
            df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            # Calculate returns for different horizons
            if horizon == HorizonType.DAILY:
                df['actual_return'] = df.groupby('symbol')['close_price'].pct_change()
                df['days_to_target'] = 1
            elif horizon == HorizonType.WEEKLY:
                df['actual_return'] = df.groupby('symbol')['close_price'].pct_change(5)
                df['days_to_target'] = 5
            else:  # MONTHLY
                df['actual_return'] = df.groupby('symbol')['close_price'].pct_change(20)
                df['days_to_target'] = 20
            
            # Determine actual direction
            df['actual_direction'] = 'neutral'
            df.loc[df['actual_return'] >= 0.03, 'actual_direction'] = 'up'
            df.loc[df['actual_return'] <= -0.03, 'actual_direction'] = 'down'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to calculate actual returns: {e}")
            return df
    
    def get_prediction_accuracy(self, predicted_probs: Dict[str, float], actual_direction: str) -> float:
        """Calculate how accurate the prediction was."""
        try:
            if actual_direction == 'up':
                return predicted_probs.get('up', 0.0)
            elif actual_direction == 'down':
                return predicted_probs.get('down', 0.0)
            else:  # neutral
                return predicted_probs.get('neutral', 0.0)
        except Exception as e:
            self.logger.error(f"Failed to calculate prediction accuracy: {e}")
            return 0.0
    
    def is_prediction_correct(self, predicted_probs: Dict[str, float], actual_direction: str) -> bool:
        """Check if prediction was correct."""
        try:
            predicted_direction = max(predicted_probs, key=predicted_probs.get)
            return predicted_direction == actual_direction
        except Exception as e:
            self.logger.error(f"Failed to check prediction correctness: {e}")
            return False
    
    def backtest_single_symbol(self, symbol: str, test_dates: List[str], 
                              horizon: HorizonType) -> List[BacktestResult]:
        """Backtest a single symbol for given dates."""
        try:
            results = []
            
            for test_date in test_dates:
                try:
                    # Get prediction for this date
                    request = PredictionRequest(
                        symbol=symbol,
                        horizon=horizon,
                        include_explanations=False
                    )
                    
                    # Simulate prediction as of test_date
                    prediction = self.fno_engine.predict_probability(request)
                    
                    # Get actual outcome
                    actual_data = self.get_actual_outcome(symbol, test_date, horizon)
                    
                    if actual_data is not None:
                        result = BacktestResult(
                            symbol=symbol,
                            prediction_date=test_date,
                            horizon=horizon.value,
                            predicted_up_prob=prediction.up_probability,
                            predicted_down_prob=prediction.down_probability,
                            predicted_neutral_prob=prediction.neutral_probability,
                            confidence_score=prediction.confidence_score,
                            actual_return=actual_data['return'],
                            actual_direction=actual_data['direction'],
                            prediction_correct=self.is_prediction_correct({
                                'up': prediction.up_probability,
                                'down': prediction.down_probability,
                                'neutral': prediction.neutral_probability
                            }, actual_data['direction']),
                            prediction_accuracy=self.get_prediction_accuracy({
                                'up': prediction.up_probability,
                                'down': prediction.down_probability,
                                'neutral': prediction.neutral_probability
                            }, actual_data['direction']),
                            days_to_target=actual_data['days_to_target']
                        )
                        results.append(result)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to backtest {symbol} for {test_date}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to backtest symbol {symbol}: {e}")
            return []
    
    def get_actual_outcome(self, symbol: str, prediction_date: str, 
                          horizon: HorizonType) -> Optional[Dict[str, Any]]:
        """Get actual outcome for a prediction."""
        try:
            # Calculate target date based on horizon
            pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
            
            if horizon == HorizonType.DAILY:
                target_date = pred_date + timedelta(days=1)
                days_to_target = 1
            elif horizon == HorizonType.WEEKLY:
                target_date = pred_date + timedelta(days=5)
                days_to_target = 5
            else:  # MONTHLY
                target_date = pred_date + timedelta(days=20)
                days_to_target = 20
            
            # Get price data using parameterized query
            query = """
                SELECT 
                    TckrSymb as symbol,
                    TRADE_DATE as date,
                    ClsPric as close_price
                FROM fno_bhav_copy
                WHERE TckrSymb = ?
                AND TRADE_DATE IN (?, ?)
                ORDER BY TRADE_DATE
            """
            
            df = self.db_manager.connection.execute(query, [symbol, prediction_date, target_date.strftime('%Y-%m-%d')]).fetchdf()
            
            if len(df) >= 2:
                # Get prediction date price
                pred_data = df[df['date'] == prediction_date]
                target_data = df[df['date'] == target_date.strftime('%Y-%m-%d')]
                
                if len(pred_data) > 0 and len(target_data) > 0:
                    pred_price = pred_data['close_price'].iloc[0]
                    target_price = target_data['close_price'].iloc[0]
                    
                    actual_return = (target_price - pred_price) / pred_price
                    
                    # Determine direction
                    if actual_return >= 0.03:
                        direction = 'up'
                    elif actual_return <= -0.03:
                        direction = 'down'
                    else:
                        direction = 'neutral'
                    
                    return {
                        'return': actual_return,
                        'direction': direction,
                        'days_to_target': days_to_target
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get actual outcome: {e}")
            return None
    
    def backtest_all_symbols(self, symbols: List[str], test_dates: List[str], 
                            horizon: HorizonType) -> List[BacktestResult]:
        """Backtest all symbols for given dates."""
        try:
            all_results = []
            
            for symbol in symbols:
                self.logger.info(f"Backtesting {symbol} for {horizon.value} horizon...")
                symbol_results = self.backtest_single_symbol(symbol, test_dates, horizon)
                all_results.extend(symbol_results)
                
                # Progress update
                self.logger.info(f"Completed {symbol}: {len(symbol_results)} predictions")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Failed to backtest all symbols: {e}")
            return []
    
    def generate_test_dates(self, start_date: str, end_date: str, 
                           num_dates: int = 10) -> List[str]:
        """Generate random test dates within the specified range."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate all possible dates
            all_dates = []
            current = start
            while current <= end:
                all_dates.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)
            
            # Randomly select dates
            if len(all_dates) <= num_dates:
                return all_dates
            else:
                selected_dates = np.random.choice(all_dates, num_dates, replace=False)
                return sorted(selected_dates.tolist())
                
        except Exception as e:
            self.logger.error(f"Failed to generate test dates: {e}")
            return []
    
    def calculate_summary_stats(self, results: List[BacktestResult]) -> BacktestSummary:
        """Calculate summary statistics from backtest results."""
        try:
            if not results:
                return BacktestSummary(
                    total_predictions=0,
                    correct_predictions=0,
                    accuracy=0.0,
                    avg_confidence=0.0,
                    avg_prediction_accuracy=0.0,
                    symbol_accuracy={},
                    horizon_accuracy={},
                    date_range=('', ''),
                    test_dates=[]
                )
            
            # Basic stats
            total_predictions = len(results)
            correct_predictions = sum(1 for r in results if r.prediction_correct)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            avg_confidence = np.mean([r.confidence_score for r in results])
            avg_prediction_accuracy = np.mean([r.prediction_accuracy for r in results])
            
            # Symbol accuracy
            symbol_accuracy = {}
            for symbol in set(r.symbol for r in results):
                symbol_results = [r for r in results if r.symbol == symbol]
                symbol_correct = sum(1 for r in symbol_results if r.prediction_correct)
                symbol_accuracy[symbol] = symbol_correct / len(symbol_results) if symbol_results else 0.0
            
            # Horizon accuracy
            horizon_accuracy = {}
            for horizon in set(r.horizon for r in results):
                horizon_results = [r for r in results if r.horizon == horizon]
                horizon_correct = sum(1 for r in horizon_results if r.prediction_correct)
                horizon_accuracy[horizon] = horizon_correct / len(horizon_results) if horizon_results else 0.0
            
            # Date range
            dates = [r.prediction_date for r in results]
            date_range = (min(dates), max(dates)) if dates else ('', '')
            test_dates = sorted(list(set(dates)))
            
            return BacktestSummary(
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
                avg_confidence=avg_confidence,
                avg_prediction_accuracy=avg_prediction_accuracy,
                symbol_accuracy=symbol_accuracy,
                horizon_accuracy=horizon_accuracy,
                date_range=date_range,
                test_dates=test_dates
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate summary stats: {e}")
            return BacktestSummary(
                total_predictions=0,
                correct_predictions=0,
                accuracy=0.0,
                avg_confidence=0.0,
                avg_prediction_accuracy=0.0,
                symbol_accuracy={},
                horizon_accuracy={},
                date_range=('', ''),
                test_dates=[]
            )
    
    def save_results(self, results: List[BacktestResult], summary: BacktestSummary, 
                    filename: str):
        """Save backtest results to CSV and JSON."""
        try:
            # Save detailed results to CSV
            results_df = pd.DataFrame([asdict(r) for r in results])
            csv_path = self.results_dir / f"{filename}_detailed.csv"
            results_df.to_csv(csv_path, index=False)
            self.logger.info(f"Detailed results saved to {csv_path}")
            
            # Save summary to JSON
            summary_dict = asdict(summary)
            json_path = self.results_dir / f"{filename}_summary.json"
            with open(json_path, 'w') as f:
                json.dump(summary_dict, f, indent=2)
            self.logger.info(f"Summary saved to {json_path}")
            
            # Save symbol-wise results
            symbol_results = {}
            for symbol in set(r.symbol for r in results):
                symbol_data = [r for r in results if r.symbol == symbol]
                symbol_df = pd.DataFrame([asdict(r) for r in symbol_data])
                symbol_path = self.results_dir / f"{filename}_{symbol}.csv"
                symbol_df.to_csv(symbol_path, index=False)
                symbol_results[symbol] = len(symbol_data)
            
            # Save symbol summary
            symbol_summary_path = self.results_dir / f"{filename}_symbols.json"
            with open(symbol_summary_path, 'w') as f:
                json.dump(symbol_results, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def run_comprehensive_backtest(self, symbols: List[str], start_date: str, 
                                  end_date: str, num_test_dates: int = 10) -> Dict[str, Any]:
        """Run comprehensive backtest for all horizons."""
        try:
            self.logger.info("Starting comprehensive backtest...")
            
            # Generate test dates
            test_dates = self.generate_test_dates(start_date, end_date, num_test_dates)
            self.logger.info(f"Generated {len(test_dates)} test dates: {test_dates}")
            
            all_results = {}
            all_summaries = {}
            
            # Test all horizons
            for horizon in [HorizonType.DAILY, HorizonType.WEEKLY, HorizonType.MONTHLY]:
                self.logger.info(f"Testing {horizon.value} horizon...")
                
                # Run backtest
                results = self.backtest_all_symbols(symbols, test_dates, horizon)
                summary = self.calculate_summary_stats(results)
                
                all_results[horizon.value] = results
                all_summaries[horizon.value] = summary
                
                # Save results
                filename = f"backtest_{horizon.value}_{start_date}_to_{end_date}"
                self.save_results(results, summary, filename)
                
                self.logger.info(f"{horizon.value} results: {summary.total_predictions} predictions, "
                               f"{summary.accuracy:.2%} accuracy")
            
            # Create overall summary
            overall_summary = {
                'test_period': f"{start_date} to {end_date}",
                'test_dates': test_dates,
                'symbols_tested': len(symbols),
                'horizons': {}
            }
            
            for horizon, summary in all_summaries.items():
                overall_summary['horizons'][horizon] = {
                    'total_predictions': summary.total_predictions,
                    'accuracy': summary.accuracy,
                    'avg_confidence': summary.avg_confidence,
                    'avg_prediction_accuracy': summary.avg_prediction_accuracy
                }
            
            # Save overall summary
            overall_filename = f"backtest_overall_{start_date}_to_{end_date}"
            overall_path = self.results_dir / f"{overall_filename}.json"
            with open(overall_path, 'w') as f:
                json.dump(overall_summary, f, indent=2)
            
            self.logger.info("Comprehensive backtest completed!")
            return overall_summary
            
        except Exception as e:
            self.logger.error(f"Failed to run comprehensive backtest: {e}")
            return {}
