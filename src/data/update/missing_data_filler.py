"""
Missing Data Filler for AI Hedge Fund.
Intelligently fills missing data using various interpolation and estimation methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import yfinance as yf
from scipy import interpolate

from ..database.duckdb_manager import DatabaseManager
from ..collectors.async_data_collector import AsyncDataCollector
from ..database.models import DataCollectionConfig


class MissingDataFiller:
    """Handles intelligent filling of missing data points."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize missing data filler.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.collector = AsyncDataCollector(self.db_manager)
        
        # Interpolation methods for different data types
        self.interpolation_methods = {
            'price': self._interpolate_price_data,
            'volume': self._interpolate_volume_data,
            'fundamental': self._interpolate_fundamental_data,
            'technical_indicators': self._interpolate_indicators
        }
    
    async def fill_missing_data(self, ticker: str, start_date: str, end_date: str, 
                               fill_method: str = "smart") -> Dict[str, Any]:
        """
        Fill missing data for a ticker in the specified date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fill_method: Method to use ('smart', 'interpolation', 'api_retry', 'hybrid')
            
        Returns:
            Dictionary with fill results
        """
        try:
            logger.info(f"Filling missing data for {ticker} from {start_date} to {end_date} using {fill_method}")
            
            # Identify missing data
            missing_analysis = await self._analyze_missing_data(ticker, start_date, end_date)
            
            if not missing_analysis["has_missing_data"]:
                return {
                    "success": True,
                    "ticker": ticker,
                    "message": "No missing data found",
                    "filled_records": 0
                }
            
            # Choose fill strategy based on method
            if fill_method == "smart":
                result = await self._smart_fill(ticker, missing_analysis)
            elif fill_method == "interpolation":
                result = await self._interpolation_fill(ticker, missing_analysis)
            elif fill_method == "api_retry":
                result = await self._api_retry_fill(ticker, missing_analysis)
            elif fill_method == "hybrid":
                result = await self._hybrid_fill(ticker, missing_analysis)
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fill missing data for {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_missing_data(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        try:
            # Get existing technical data
            technical_df = self.db_manager.get_technical_data(ticker, start_date, end_date)
            
            # Get expected trading days
            expected_dates = self._get_trading_days(start_date, end_date)
            
            # Find missing dates
            existing_dates = []
            if not technical_df.empty and 'trade_date' in technical_df.columns:
                existing_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) 
                                for d in technical_df['trade_date']]
            
            missing_dates = [d for d in expected_dates if d not in existing_dates]
            
            # Analyze missing patterns
            missing_analysis = {
                "has_missing_data": len(missing_dates) > 0,
                "total_expected": len(expected_dates),
                "total_existing": len(existing_dates),
                "total_missing": len(missing_dates),
                "completeness_ratio": len(existing_dates) / len(expected_dates) if expected_dates else 0,
                "missing_dates": missing_dates,
                "existing_data": technical_df,
                "missing_patterns": self._analyze_missing_patterns(missing_dates, expected_dates)
            }
            
            return missing_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze missing data for {ticker}: {e}")
            return {"has_missing_data": False, "error": str(e)}
    
    def _analyze_missing_patterns(self, missing_dates: List[str], all_dates: List[str]) -> Dict[str, Any]:
        """Analyze patterns in missing data."""
        try:
            if not missing_dates:
                return {"pattern_type": "none", "consecutive_gaps": []}
            
            # Convert to datetime for analysis
            missing_dt = [datetime.strptime(d, '%Y-%m-%d') for d in missing_dates]
            missing_dt.sort()
            
            # Find consecutive gaps
            consecutive_gaps = []
            current_gap = [missing_dt[0]]
            
            for i in range(1, len(missing_dt)):
                if (missing_dt[i] - missing_dt[i-1]).days == 1:
                    current_gap.append(missing_dt[i])
                else:
                    if len(current_gap) > 1:
                        consecutive_gaps.append(current_gap)
                    current_gap = [missing_dt[i]]
            
            if len(current_gap) > 1:
                consecutive_gaps.append(current_gap)
            
            # Determine pattern type
            if len(missing_dates) == len(all_dates):
                pattern_type = "complete_missing"
            elif len(consecutive_gaps) == 0:
                pattern_type = "scattered"
            elif len(consecutive_gaps) == 1 and len(consecutive_gaps[0]) == len(missing_dates):
                pattern_type = "single_gap"
            else:
                pattern_type = "multiple_gaps"
            
            return {
                "pattern_type": pattern_type,
                "consecutive_gaps": [[d.strftime('%Y-%m-%d') for d in gap] for gap in consecutive_gaps],
                "largest_gap_size": max([len(gap) for gap in consecutive_gaps]) if consecutive_gaps else 1,
                "total_gaps": len(consecutive_gaps)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze missing patterns: {e}")
            return {"pattern_type": "unknown", "consecutive_gaps": []}
    
    async def _smart_fill(self, ticker: str, missing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Smart fill strategy - chooses best method based on data patterns."""
        try:
            pattern_type = missing_analysis["missing_patterns"]["pattern_type"]
            total_missing = missing_analysis["total_missing"]
            
            # Strategy selection based on patterns
            if pattern_type == "complete_missing":
                # No existing data, use API retry
                return await self._api_retry_fill(ticker, missing_analysis)
            elif pattern_type == "single_gap" and total_missing <= 5:
                # Small single gap, use interpolation
                return await self._interpolation_fill(ticker, missing_analysis)
            elif pattern_type == "scattered" and total_missing <= 10:
                # Scattered missing points, try API first then interpolation
                return await self._hybrid_fill(ticker, missing_analysis)
            else:
                # Large gaps or complex patterns, use API retry
                return await self._api_retry_fill(ticker, missing_analysis)
                
        except Exception as e:
            logger.error(f"Smart fill failed for {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _interpolation_fill(self, ticker: str, missing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing data using interpolation methods."""
        try:
            existing_df = missing_analysis["existing_data"]
            missing_dates = missing_analysis["missing_dates"]
            
            if existing_df.empty or len(missing_dates) == 0:
                return {"success": False, "message": "No data to interpolate"}
            
            filled_records = 0
            interpolated_data = []
            
            # Sort existing data by date
            if 'trade_date' in existing_df.columns:
                existing_df = existing_df.sort_values('trade_date')
            
            # Interpolate price data
            price_data = self._interpolate_price_data(existing_df, missing_dates)
            if price_data:
                interpolated_data.extend(price_data)
                filled_records += len(price_data)
            
            # Store interpolated data
            if interpolated_data:
                interpolated_df = pd.DataFrame(interpolated_data)
                # Store in database (would implement actual storage)
                logger.info(f"Generated {filled_records} interpolated records for {ticker}")
            
            return {
                "success": True,
                "ticker": ticker,
                "method": "interpolation",
                "filled_records": filled_records,
                "interpolated_dates": missing_dates
            }
            
        except Exception as e:
            logger.error(f"Interpolation fill failed for {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _api_retry_fill(self, ticker: str, missing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing data by retrying API calls."""
        try:
            missing_dates = missing_analysis["missing_dates"]
            
            if not missing_dates:
                return {"success": True, "filled_records": 0}
            
            # Group consecutive dates for efficient API calls
            date_ranges = self._group_consecutive_dates(missing_dates)
            
            total_filled = 0
            
            for start_date, end_date in date_ranges:
                # Create collection config
                config = DataCollectionConfig(
                    ticker=ticker,
                    start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
                    end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
                    data_types=["technical"],
                    max_workers=2,
                    retry_attempts=3
                )
                
                # Collect data for this range
                results = await self.collector.collect_historical_data(config)
                
                # Count successful records
                for result in results:
                    if result.success:
                        total_filled += result.records_collected
            
            return {
                "success": True,
                "ticker": ticker,
                "method": "api_retry",
                "filled_records": total_filled,
                "date_ranges_processed": len(date_ranges)
            }
            
        except Exception as e:
            logger.error(f"API retry fill failed for {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _hybrid_fill(self, ticker: str, missing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid fill strategy - tries API first, then interpolation."""
        try:
            # First try API retry
            api_result = await self._api_retry_fill(ticker, missing_analysis)
            
            # If API got some data, check what's still missing
            if api_result["success"] and api_result["filled_records"] > 0:
                # Re-analyze what's still missing
                updated_analysis = await self._analyze_missing_data(
                    ticker, 
                    min(missing_analysis["missing_dates"]),
                    max(missing_analysis["missing_dates"])
                )
                
                # If there's still missing data, try interpolation
                if updated_analysis["has_missing_data"]:
                    interp_result = await self._interpolation_fill(ticker, updated_analysis)
                    
                    return {
                        "success": True,
                        "ticker": ticker,
                        "method": "hybrid",
                        "filled_records": api_result["filled_records"] + interp_result.get("filled_records", 0),
                        "api_filled": api_result["filled_records"],
                        "interpolation_filled": interp_result.get("filled_records", 0)
                    }
                else:
                    return {
                        "success": True,
                        "ticker": ticker,
                        "method": "hybrid",
                        "filled_records": api_result["filled_records"],
                        "api_filled": api_result["filled_records"],
                        "interpolation_filled": 0
                    }
            else:
                # API failed, try interpolation only
                return await self._interpolation_fill(ticker, missing_analysis)
                
        except Exception as e:
            logger.error(f"Hybrid fill failed for {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    def _interpolate_price_data(self, df: pd.DataFrame, missing_dates: List[str]) -> List[Dict[str, Any]]:
        """Interpolate price data for missing dates."""
        try:
            if df.empty or not missing_dates:
                return []
            
            interpolated_records = []
            
            # Ensure we have the required columns
            required_cols = ['trade_date', 'open_price', 'high_price', 'low_price', 'close_price']
            if not all(col in df.columns for col in required_cols):
                logger.warning("Missing required columns for price interpolation")
                return []
            
            # Convert dates to datetime
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            
            # Create a complete date range
            full_dates = pd.date_range(df['trade_date'].min(), df['trade_date'].max(), freq='D')
            full_dates = [d for d in full_dates if d.weekday() < 5]  # Only weekdays
            
            # Create full dataframe with missing dates
            full_df = pd.DataFrame({'trade_date': full_dates})
            full_df = full_df.merge(df, on='trade_date', how='left')
            
            # Interpolate price columns
            price_cols = ['open_price', 'high_price', 'low_price', 'close_price']
            
            for col in price_cols:
                if col in full_df.columns:
                    # Use linear interpolation for prices
                    full_df[col] = full_df[col].interpolate(method='linear')
            
            # Interpolate volume (use forward fill then backward fill)
            if 'volume' in full_df.columns:
                full_df['volume'] = full_df['volume'].fillna(method='ffill').fillna(method='bfill')
                # Ensure volume is integer
                full_df['volume'] = full_df['volume'].fillna(0).astype(int)
            
            # Extract only the missing dates
            missing_dt = [datetime.strptime(d, '%Y-%m-%d') for d in missing_dates]
            missing_df = full_df[full_df['trade_date'].isin(missing_dt)]
            
            # Convert to records
            for _, row in missing_df.iterrows():
                if not pd.isna(row['close_price']):  # Only add if interpolation worked
                    record = {
                        'ticker': df['ticker'].iloc[0] if 'ticker' in df.columns else 'UNKNOWN',
                        'trade_date': row['trade_date'].date(),
                        'open_price': float(row['open_price']),
                        'high_price': float(row['high_price']),
                        'low_price': float(row['low_price']),
                        'close_price': float(row['close_price']),
                        'volume': int(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else 0,
                        'created_at': datetime.now(),
                        'updated_at': datetime.now()
                    }
                    interpolated_records.append(record)
            
            return interpolated_records
            
        except Exception as e:
            logger.error(f"Failed to interpolate price data: {e}")
            return []
    
    def _interpolate_volume_data(self, df: pd.DataFrame, missing_dates: List[str]) -> List[Dict[str, Any]]:
        """Interpolate volume data using historical patterns."""
        try:
            # Use average volume for missing dates
            if 'volume' in df.columns:
                avg_volume = df['volume'].mean()
                return [{'volume': int(avg_volume)} for _ in missing_dates]
            return []
        except Exception as e:
            logger.error(f"Failed to interpolate volume data: {e}")
            return []
    
    def _interpolate_fundamental_data(self, df: pd.DataFrame, missing_dates: List[str]) -> List[Dict[str, Any]]:
        """Interpolate fundamental data (usually forward fill)."""
        try:
            # Fundamental data is typically forward-filled
            if df.empty:
                return []
            
            # Get the most recent fundamental data
            latest_fundamental = df.iloc[-1].to_dict()
            
            # Apply to missing dates
            interpolated_records = []
            for date_str in missing_dates:
                record = latest_fundamental.copy()
                record['report_date'] = datetime.strptime(date_str, '%Y-%m-%d').date()
                record['updated_at'] = datetime.now()
                interpolated_records.append(record)
            
            return interpolated_records
            
        except Exception as e:
            logger.error(f"Failed to interpolate fundamental data: {e}")
            return []
    
    def _interpolate_indicators(self, df: pd.DataFrame, missing_dates: List[str]) -> List[Dict[str, Any]]:
        """Interpolate technical indicators."""
        try:
            # Technical indicators can be recalculated based on price data
            # For now, use linear interpolation
            indicator_cols = ['sma_20', 'sma_50', 'rsi_14', 'macd']
            
            interpolated_records = []
            for col in indicator_cols:
                if col in df.columns:
                    # Simple linear interpolation for indicators
                    interpolated_values = df[col].interpolate(method='linear')
                    # Add to records...
            
            return interpolated_records
            
        except Exception as e:
            logger.error(f"Failed to interpolate indicators: {e}")
            return []
    
    def _get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get list of trading days between start and end date."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            trading_days = []
            current = start
            
            while current <= end:
                # Skip weekends (Monday = 0, Friday = 4)
                if current.weekday() < 5:
                    trading_days.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)
            
            return trading_days
            
        except Exception as e:
            logger.error(f"Failed to get trading days: {e}")
            return []
    
    def _group_consecutive_dates(self, dates: List[str]) -> List[Tuple[str, str]]:
        """Group consecutive dates into ranges for efficient API calls."""
        try:
            if not dates:
                return []
            
            # Sort dates
            sorted_dates = sorted([datetime.strptime(d, '%Y-%m-%d') for d in dates])
            
            ranges = []
            start_date = sorted_dates[0]
            end_date = sorted_dates[0]
            
            for i in range(1, len(sorted_dates)):
                if (sorted_dates[i] - end_date).days <= 1:
                    # Consecutive or adjacent date
                    end_date = sorted_dates[i]
                else:
                    # Gap found, close current range and start new one
                    ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                    start_date = sorted_dates[i]
                    end_date = sorted_dates[i]
            
            # Add the last range
            ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            return ranges
            
        except Exception as e:
            logger.error(f"Failed to group consecutive dates: {e}")
            return [(dates[0], dates[-1])] if dates else []