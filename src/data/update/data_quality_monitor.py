"""
Data Quality Monitor for AI Hedge Fund.
Monitors and tracks data quality metrics across all data sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from pathlib import Path
import json

from ..database.duckdb_manager import DatabaseManager
from ..database.models import DataQualityMetrics


class DataQualityMonitor:
    """Monitors and tracks data quality metrics."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize data quality monitor.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
    
    async def update_quality_metrics(self, ticker: str, target_date: str) -> Dict[str, Any]:
        """
        Update quality metrics for a ticker on a specific date.
        
        Args:
            ticker: Stock ticker symbol
            target_date: Date to analyze (YYYY-MM-DD)
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            logger.info(f"Updating quality metrics for {ticker} on {target_date}")
            
            # Analyze technical data quality
            technical_metrics = await self._analyze_technical_quality(ticker, target_date)
            
            # Analyze fundamental data quality
            fundamental_metrics = await self._analyze_fundamental_quality(ticker, target_date)
            
            # Store quality metrics in database
            if technical_metrics:
                await self._store_quality_metrics(ticker, target_date, "technical", technical_metrics)
            
            if fundamental_metrics:
                await self._store_quality_metrics(ticker, target_date, "fundamental", fundamental_metrics)
            
            return {
                "success": True,
                "ticker": ticker,
                "date": target_date,
                "technical_metrics": technical_metrics,
                "fundamental_metrics": fundamental_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to update quality metrics for {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_technical_quality(self, ticker: str, target_date: str) -> Optional[Dict[str, float]]:
        """Analyze technical data quality."""
        try:
            # Get technical data for the last 30 days
            start_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            df = self.db_manager.get_technical_data(ticker, start_date, target_date)
            
            if df.empty:
                return None
            
            # Calculate quality metrics
            total_expected_days = self._count_trading_days(start_date, target_date)
            actual_days = len(df)
            
            # Completeness score
            completeness_score = (actual_days / total_expected_days) * 100 if total_expected_days > 0 else 0
            
            # Accuracy score (check for data anomalies)
            accuracy_issues = 0
            
            # Check for missing prices
            price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
            for col in price_columns:
                if col in df.columns:
                    missing_prices = df[col].isnull().sum()
                    accuracy_issues += missing_prices
            
            # Check for zero or negative volumes
            if 'volume' in df.columns:
                invalid_volume = (df['volume'] <= 0).sum()
                accuracy_issues += invalid_volume
            
            # Check for price inconsistencies (high < low, etc.)
            if all(col in df.columns for col in ['high_price', 'low_price']):
                price_inconsistencies = (df['high_price'] < df['low_price']).sum()
                accuracy_issues += price_inconsistencies
            
            total_data_points = len(df) * len([col for col in price_columns if col in df.columns])
            accuracy_score = max(0, 100 - (accuracy_issues / total_data_points * 100)) if total_data_points > 0 else 0
            
            # Timeliness score (how recent is the latest data)
            latest_date = df['trade_date'].max() if 'trade_date' in df.columns else None
            if latest_date:
                if isinstance(latest_date, str):
                    latest_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
                elif hasattr(latest_date, 'date'):
                    latest_date = latest_date.date()
                
                target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                days_behind = (target_date_obj - latest_date).days
                timeliness_score = max(0, 100 - (days_behind * 10))  # 10% penalty per day behind
            else:
                timeliness_score = 0
            
            # Consistency score (check for data patterns)
            consistency_score = 100  # Start with perfect score
            
            # Check for duplicate dates
            if 'trade_date' in df.columns:
                duplicates = df['trade_date'].duplicated().sum()
                consistency_score -= (duplicates / len(df)) * 50
            
            # Check for extreme price movements (possible errors)
            if 'close_price' in df.columns and len(df) > 1:
                price_changes = df['close_price'].pct_change().abs()
                extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
                consistency_score -= (extreme_changes / len(df)) * 25
            
            consistency_score = max(0, consistency_score)
            
            return {
                "completeness_score": round(completeness_score, 2),
                "accuracy_score": round(accuracy_score, 2),
                "timeliness_score": round(timeliness_score, 2),
                "consistency_score": round(consistency_score, 2),
                "total_records": actual_days,
                "missing_records": max(0, total_expected_days - actual_days),
                "error_count": accuracy_issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze technical quality for {ticker}: {e}")
            return None
    
    async def _analyze_fundamental_quality(self, ticker: str, target_date: str) -> Optional[Dict[str, float]]:
        """Analyze fundamental data quality."""
        try:
            # Get fundamental data for the last year
            start_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
            df = self.db_manager.get_fundamental_data(ticker, start_date, target_date)
            
            if df.empty:
                return None
            
            # Calculate quality metrics for fundamental data
            total_expected_reports = 4  # Quarterly reports in a year
            actual_reports = len(df)
            
            # Completeness score
            completeness_score = min(100, (actual_reports / total_expected_reports) * 100)
            
            # Accuracy score (check for missing key metrics)
            accuracy_issues = 0
            key_metrics = ['revenue', 'net_income', 'total_assets', 'total_equity']
            
            for metric in key_metrics:
                if metric in df.columns:
                    missing_values = df[metric].isnull().sum()
                    accuracy_issues += missing_values
            
            total_data_points = len(df) * len(key_metrics)
            accuracy_score = max(0, 100 - (accuracy_issues / total_data_points * 100)) if total_data_points > 0 else 0
            
            # Timeliness score (fundamental data is typically quarterly)
            latest_date = df['report_date'].max() if 'report_date' in df.columns else None
            if latest_date:
                if isinstance(latest_date, str):
                    latest_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
                elif hasattr(latest_date, 'date'):
                    latest_date = latest_date.date()
                
                target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                days_behind = (target_date_obj - latest_date).days
                # Fundamental data can be up to 90 days behind and still be considered timely
                timeliness_score = max(0, 100 - max(0, (days_behind - 90) * 2))
            else:
                timeliness_score = 0
            
            # Consistency score
            consistency_score = 100
            
            # Check for negative values in key metrics that should be positive
            if 'revenue' in df.columns:
                negative_revenue = (df['revenue'] < 0).sum()
                consistency_score -= (negative_revenue / len(df)) * 30
            
            if 'total_assets' in df.columns:
                negative_assets = (df['total_assets'] < 0).sum()
                consistency_score -= (negative_assets / len(df)) * 30
            
            consistency_score = max(0, consistency_score)
            
            return {
                "completeness_score": round(completeness_score, 2),
                "accuracy_score": round(accuracy_score, 2),
                "timeliness_score": round(timeliness_score, 2),
                "consistency_score": round(consistency_score, 2),
                "total_records": actual_reports,
                "missing_records": max(0, total_expected_reports - actual_reports),
                "error_count": accuracy_issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze fundamental quality for {ticker}: {e}")
            return None
    
    async def _store_quality_metrics(self, ticker: str, target_date: str, data_type: str, metrics: Dict[str, float]):
        """Store quality metrics in database."""
        try:
            # Create quality metrics object
            quality_metrics = DataQualityMetrics(
                ticker=ticker,
                quality_date=datetime.strptime(target_date, '%Y-%m-%d').date(),
                data_type=data_type,
                completeness_score=metrics["completeness_score"],
                accuracy_score=metrics["accuracy_score"],
                timeliness_score=metrics["timeliness_score"],
                consistency_score=metrics["consistency_score"],
                total_records=metrics["total_records"],
                missing_records=metrics["missing_records"],
                error_count=metrics["error_count"]
            )
            
            # Convert to DataFrame and store
            df = pd.DataFrame([quality_metrics.dict()])
            
            # Store in database (you would implement this in the database manager)
            # For now, we'll log the metrics
            logger.info(f"Quality metrics for {ticker} ({data_type}): "
                       f"Overall Score: {(metrics['completeness_score'] + metrics['accuracy_score'] + metrics['timeliness_score'] + metrics['consistency_score']) / 4:.1f}")
            
        except Exception as e:
            logger.error(f"Failed to store quality metrics: {e}")
    
    def _count_trading_days(self, start_date: str, end_date: str) -> int:
        """Count trading days between two dates (excluding weekends)."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            trading_days = 0
            current = start
            
            while current <= end:
                # Skip weekends (Monday = 0, Friday = 4)
                if current.weekday() < 5:
                    trading_days += 1
                current += timedelta(days=1)
            
            return trading_days
            
        except Exception as e:
            logger.error(f"Failed to count trading days: {e}")
            return 0
    
    async def generate_quality_report(self, tickers: List[str], date_range: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            date_range: Number of days to analyze
            
        Returns:
            Comprehensive quality report
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=date_range)).strftime('%Y-%m-%d')
            
            report = {
                "generated_at": datetime.now().isoformat(),
                "analysis_period": f"{start_date} to {end_date}",
                "total_tickers": len(tickers),
                "ticker_quality": {},
                "summary": {
                    "average_completeness": 0,
                    "average_accuracy": 0,
                    "average_timeliness": 0,
                    "average_consistency": 0,
                    "tickers_with_issues": 0,
                    "total_missing_days": 0
                }
            }
            
            all_scores = {"completeness": [], "accuracy": [], "timeliness": [], "consistency": []}
            total_missing = 0
            issues_count = 0
            
            for ticker in tickers:
                # Analyze technical quality
                tech_metrics = await self._analyze_technical_quality(ticker, end_date)
                
                # Analyze fundamental quality
                fund_metrics = await self._analyze_fundamental_quality(ticker, end_date)
                
                ticker_report = {
                    "technical": tech_metrics,
                    "fundamental": fund_metrics,
                    "overall_score": 0,
                    "has_issues": False
                }
                
                # Calculate overall score
                scores = []
                if tech_metrics:
                    tech_overall = (tech_metrics["completeness_score"] + tech_metrics["accuracy_score"] + 
                                  tech_metrics["timeliness_score"] + tech_metrics["consistency_score"]) / 4
                    scores.append(tech_overall)
                    all_scores["completeness"].append(tech_metrics["completeness_score"])
                    all_scores["accuracy"].append(tech_metrics["accuracy_score"])
                    all_scores["timeliness"].append(tech_metrics["timeliness_score"])
                    all_scores["consistency"].append(tech_metrics["consistency_score"])
                    total_missing += tech_metrics["missing_records"]
                
                if fund_metrics:
                    fund_overall = (fund_metrics["completeness_score"] + fund_metrics["accuracy_score"] + 
                                  fund_metrics["timeliness_score"] + fund_metrics["consistency_score"]) / 4
                    scores.append(fund_overall)
                
                if scores:
                    ticker_report["overall_score"] = sum(scores) / len(scores)
                    ticker_report["has_issues"] = ticker_report["overall_score"] < 80
                    
                    if ticker_report["has_issues"]:
                        issues_count += 1
                
                report["ticker_quality"][ticker] = ticker_report
            
            # Calculate summary statistics
            if all_scores["completeness"]:
                report["summary"]["average_completeness"] = sum(all_scores["completeness"]) / len(all_scores["completeness"])
                report["summary"]["average_accuracy"] = sum(all_scores["accuracy"]) / len(all_scores["accuracy"])
                report["summary"]["average_timeliness"] = sum(all_scores["timeliness"]) / len(all_scores["timeliness"])
                report["summary"]["average_consistency"] = sum(all_scores["consistency"]) / len(all_scores["consistency"])
            
            report["summary"]["tickers_with_issues"] = issues_count
            report["summary"]["total_missing_days"] = total_missing
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {"error": str(e)}
    
    async def get_data_alerts(self, threshold: float = 70.0) -> List[Dict[str, Any]]:
        """
        Get data quality alerts for tickers below threshold.
        
        Args:
            threshold: Quality score threshold for alerts
            
        Returns:
            List of quality alerts
        """
        try:
            alerts = []
            
            # This would typically query stored quality metrics from database
            # For now, we'll return a placeholder structure
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get data alerts: {e}")
            return []