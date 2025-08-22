"""
Daily Data Updater for AI Hedge Fund.
Handles incremental updates from the last stored date to current date.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from pathlib import Path
import json

from ..database.duckdb_manager import DatabaseManager
from ..collectors.async_data_collector import AsyncDataCollector
from ..database.models import DataCollectionConfig, DataQualityMetrics
from .data_quality_monitor import DataQualityMonitor


class DailyDataUpdater:
    """Handles daily incremental data updates for all configured tickers."""
    
    def __init__(self, db_manager: DatabaseManager = None, config_path: str = "config/tickers.json"):
        """
        Initialize daily data updater.
        
        Args:
            db_manager: Database manager instance
            config_path: Path to ticker configuration file
        """
        self.db_manager = db_manager or DatabaseManager()
        self.config_path = config_path
        self.collector = AsyncDataCollector(self.db_manager)
        self.quality_monitor = DataQualityMonitor(self.db_manager)
        
        # Load ticker configuration
        self.tickers_config = self._load_ticker_config()
    
    def _load_ticker_config(self) -> Dict[str, Any]:
        """Load ticker configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    "indian_stocks": [
                        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                        "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS"
                    ],
                    "us_stocks": [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"
                    ],
                    "data_types": ["technical", "fundamental"],
                    "update_frequency": "daily",
                    "max_workers": 5,
                    "retry_attempts": 3
                }
                
                # Create config directory and save default
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load ticker configuration: {e}")
            return {"indian_stocks": [], "us_stocks": [], "data_types": ["technical"]}
    
    async def run_daily_update(self, target_date: str = None) -> Dict[str, Any]:
        """
        Run daily data update for all configured tickers.
        
        Args:
            target_date: Target date for update (YYYY-MM-DD). If None, uses yesterday.
            
        Returns:
            Dictionary with update results and statistics
        """
        try:
            # Determine target date
            if target_date is None:
                target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"Starting daily data update for {target_date}")
            
            # Get all tickers to update
            all_tickers = (self.tickers_config.get("indian_stocks", []) + 
                          self.tickers_config.get("us_stocks", []))
            
            if not all_tickers:
                logger.warning("No tickers configured for update")
                return {"success": False, "message": "No tickers configured"}
            
            # Run updates for all tickers
            results = await self._update_all_tickers(all_tickers, target_date)
            
            # Generate quality report
            quality_report = await self._generate_quality_report(target_date)
            
            # Compile final results
            update_summary = {
                "success": True,
                "target_date": target_date,
                "total_tickers": len(all_tickers),
                "successful_updates": results["successful"],
                "failed_updates": results["failed"],
                "total_records_collected": results["total_records"],
                "duration_seconds": results["duration"],
                "quality_report": quality_report,
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Daily update completed: {results['successful']}/{len(all_tickers)} tickers successful")
            return update_summary
            
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_all_tickers(self, tickers: List[str], target_date: str) -> Dict[str, Any]:
        """Update data for all tickers."""
        start_time = datetime.now()
        successful = 0
        failed = 0
        total_records = 0
        
        # Process tickers in batches to avoid overwhelming the system
        batch_size = self.tickers_config.get("max_workers", 5)
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            # Process batch concurrently
            tasks = [self._update_single_ticker(ticker, target_date) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for ticker, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to update {ticker}: {result}")
                    failed += 1
                elif result["success"]:
                    successful += 1
                    total_records += result.get("records_collected", 0)
                    logger.info(f"✅ Updated {ticker}: {result.get('records_collected', 0)} records")
                else:
                    failed += 1
                    logger.warning(f"❌ Failed to update {ticker}: {result.get('error', 'Unknown error')}")
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "successful": successful,
            "failed": failed,
            "total_records": total_records,
            "duration": duration
        }
    
    async def _update_single_ticker(self, ticker: str, target_date: str) -> Dict[str, Any]:
        """Update data for a single ticker."""
        try:
            # Check what data needs updating
            update_info = await self._check_update_requirements(ticker, target_date)
            
            if not update_info["needs_update"]:
                return {
                    "success": True,
                    "ticker": ticker,
                    "records_collected": 0,
                    "message": "No update needed"
                }
            
            # Collect missing data
            config = DataCollectionConfig(
                ticker=ticker,
                start_date=datetime.strptime(update_info["start_date"], '%Y-%m-%d').date(),
                end_date=datetime.strptime(target_date, '%Y-%m-%d').date(),
                data_types=self.tickers_config.get("data_types", ["technical"]),
                max_workers=3,
                retry_attempts=self.tickers_config.get("retry_attempts", 3)
            )
            
            # Collect data
            results = await self.collector.collect_historical_data(config)
            
            # Store data in database
            total_records = await self._store_collected_data(ticker, results)
            
            # Update quality metrics
            await self.quality_monitor.update_quality_metrics(ticker, target_date)
            
            return {
                "success": True,
                "ticker": ticker,
                "records_collected": total_records,
                "data_types": config.data_types
            }
            
        except Exception as e:
            logger.error(f"Failed to update {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e)
            }
    
    async def _check_update_requirements(self, ticker: str, target_date: str) -> Dict[str, Any]:
        """Check what data needs to be updated for a ticker."""
        try:
            # Get latest data dates
            latest_technical = self.db_manager.get_latest_data_date(ticker, "technical")
            latest_fundamental = self.db_manager.get_latest_data_date(ticker, "fundamental")
            
            # Determine start date for update
            start_dates = []
            
            if "technical" in self.tickers_config.get("data_types", []):
                if latest_technical:
                    # Start from day after latest data
                    next_date = datetime.strptime(latest_technical, '%Y-%m-%d') + timedelta(days=1)
                    start_dates.append(next_date.strftime('%Y-%m-%d'))
                else:
                    # No data exists, start from 30 days ago
                    start_date = datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=30)
                    start_dates.append(start_date.strftime('%Y-%m-%d'))
            
            if "fundamental" in self.tickers_config.get("data_types", []):
                if latest_fundamental:
                    # Fundamental data is usually quarterly, check if we need update
                    last_update = datetime.strptime(latest_fundamental, '%Y-%m-%d')
                    if (datetime.strptime(target_date, '%Y-%m-%d') - last_update).days > 90:
                        start_dates.append(latest_fundamental)
                else:
                    # No fundamental data, start from 1 year ago
                    start_date = datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=365)
                    start_dates.append(start_date.strftime('%Y-%m-%d'))
            
            if not start_dates:
                return {"needs_update": False}
            
            # Use earliest start date
            start_date = min(start_dates)
            
            # Check if update is needed
            needs_update = start_date <= target_date
            
            return {
                "needs_update": needs_update,
                "start_date": start_date,
                "latest_technical": latest_technical,
                "latest_fundamental": latest_fundamental
            }
            
        except Exception as e:
            logger.error(f"Failed to check update requirements for {ticker}: {e}")
            return {"needs_update": False, "error": str(e)}
    
    async def _store_collected_data(self, ticker: str, results: List[Any]) -> int:
        """Store collected data in database."""
        total_records = 0
        
        try:
            for result in results:
                if result.success and result.records_collected > 0:
                    # Data is already stored by the collector
                    total_records += result.records_collected
                    logger.debug(f"Stored {result.records_collected} {result.data_type} records for {ticker}")
            
            return total_records
            
        except Exception as e:
            logger.error(f"Failed to store data for {ticker}: {e}")
            return 0
    
    async def _generate_quality_report(self, target_date: str) -> Dict[str, Any]:
        """Generate data quality report for the update."""
        try:
            all_tickers = (self.tickers_config.get("indian_stocks", []) + 
                          self.tickers_config.get("us_stocks", []))
            
            quality_scores = []
            missing_data_counts = []
            
            for ticker in all_tickers:
                # Get quality metrics
                quality_metrics = self.db_manager.get_data_quality_metrics(ticker, "technical")
                if not quality_metrics.empty:
                    latest_quality = quality_metrics.iloc[0]
                    quality_scores.append(latest_quality['completeness_score'])
                
                # Get missing data count
                missing_dates = self.db_manager.get_missing_data_dates(
                    ticker, 
                    (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d'),
                    target_date
                )
                missing_data_counts.append(len(missing_dates))
            
            return {
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "total_missing_days": sum(missing_data_counts),
                "tickers_with_issues": len([x for x in quality_scores if x < 90]) if quality_scores else 0,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {"error": str(e)}
    
    async def update_missing_data(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Update missing data for a specific ticker and date range."""
        try:
            logger.info(f"Updating missing data for {ticker} from {start_date} to {end_date}")
            
            # Get missing dates
            missing_dates = self.db_manager.get_missing_data_dates(ticker, start_date, end_date)
            
            if not missing_dates:
                return {
                    "success": True,
                    "ticker": ticker,
                    "message": "No missing data found"
                }
            
            logger.info(f"Found {len(missing_dates)} missing dates for {ticker}")
            
            # Create collection config for missing data
            config = DataCollectionConfig(
                ticker=ticker,
                start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
                end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
                data_types=self.tickers_config.get("data_types", ["technical"]),
                max_workers=3,
                retry_attempts=self.tickers_config.get("retry_attempts", 3)
            )
            
            # Collect missing data
            results = await self.collector.collect_historical_data(config)
            
            # Store data
            total_records = await self._store_collected_data(ticker, results)
            
            return {
                "success": True,
                "ticker": ticker,
                "missing_dates_found": len(missing_dates),
                "records_collected": total_records
            }
            
        except Exception as e:
            logger.error(f"Failed to update missing data for {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e)
            }
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status for all tickers."""
        try:
            all_tickers = (self.tickers_config.get("indian_stocks", []) + 
                          self.tickers_config.get("us_stocks", []))
            
            status = {
                "total_tickers": len(all_tickers),
                "ticker_status": {},
                "last_check": datetime.now().isoformat()
            }
            
            for ticker in all_tickers:
                latest_technical = self.db_manager.get_latest_data_date(ticker, "technical")
                latest_fundamental = self.db_manager.get_latest_data_date(ticker, "fundamental")
                
                # Calculate days behind
                today = datetime.now().date()
                tech_days_behind = 0
                fund_days_behind = 0
                
                if latest_technical:
                    tech_days_behind = (today - datetime.strptime(latest_technical, '%Y-%m-%d').date()).days
                
                if latest_fundamental:
                    fund_days_behind = (today - datetime.strptime(latest_fundamental, '%Y-%m-%d').date()).days
                
                status["ticker_status"][ticker] = {
                    "latest_technical": latest_technical,
                    "latest_fundamental": latest_fundamental,
                    "technical_days_behind": tech_days_behind,
                    "fundamental_days_behind": fund_days_behind,
                    "needs_update": tech_days_behind > 1 or fund_days_behind > 90
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get update status: {e}")
            return {"error": str(e)}