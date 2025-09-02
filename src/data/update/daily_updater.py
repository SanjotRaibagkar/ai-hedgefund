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
from ..downloaders.eod_extra_data_downloader import EODExtraDataDownloader


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
        self.eod_downloader = EODExtraDataDownloader(force_recreate_tables=False)
        
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
                    "eod_extra_data": {
                        "enabled": True,
                        "types": ["fno_bhav_copy", "equity_bhav_copy_delivery", "bhav_copy_indices", "fii_dii_activity"]
                    },
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
            return {
                "indian_stocks": [], 
                "us_stocks": [], 
                "data_types": ["technical"],
                "eod_extra_data": {
                    "enabled": True,
                    "types": ["fno_bhav_copy", "equity_bhav_copy_delivery", "bhav_copy_indices", "fii_dii_activity"]
                }
            }
    
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
            ticker_results = await self._update_all_tickers(all_tickers, target_date)
            
            # Run EOD extra data updates
            eod_results = await self._update_eod_extra_data(target_date)
            
            # Generate quality report
            quality_report = await self._generate_quality_report(target_date)
            
            # Combine results
            results = {
                "success": True,
                "target_date": target_date,
                "ticker_updates": ticker_results,
                "eod_extra_data": eod_results,
                "quality_report": quality_report,
                "summary": {
                    "total_tickers": len(all_tickers),
                    "successful_tickers": ticker_results.get("successful", 0),
                    "failed_tickers": ticker_results.get("failed", 0),
                    "eod_data_types": len(eod_results.get("successful_types", [])),
                    "failed_eod_types": len(eod_results.get("failed_types", []))
                }
            }
            
            logger.info(f"Daily update completed: {results['summary']['successful_tickers']}/{results['summary']['total_tickers']} tickers successful")
            logger.info(f"EOD extra data: {results['summary']['eod_data_types']} types successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_eod_extra_data(self, target_date: str) -> Dict[str, Any]:
        """Update EOD extra data for the target date."""
        try:
            if not self.tickers_config.get("eod_extra_data", {}).get("enabled", True):
                logger.info("EOD extra data updates disabled in configuration")
                return {"success": True, "message": "EOD extra data updates disabled"}
            
            logger.info(f"Starting EOD extra data update for {target_date}")
            
            eod_types = self.tickers_config.get("eod_extra_data", {}).get("types", [])
            successful_types = []
            failed_types = []
            
            # Convert target_date to DD-MM-YYYY format for NSE API
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            nse_date_format = target_dt.strftime('%d-%m-%Y')
            
            for eod_type in eod_types:
                try:
                    logger.info(f"Downloading {eod_type} for {nse_date_format}")
                    
                    if eod_type == "fno_bhav_copy":
                        df = self.eod_downloader.nse.fno_bhav_copy(nse_date_format)
                        if not df.empty:
                            df['TRADE_DATE'] = target_dt.date()
                            df['last_updated'] = datetime.now().isoformat()
                            # FIXED: Use INSERT OR REPLACE instead of DELETE + INSERT
                            # This preserves existing data and only updates duplicates
                            self.eod_downloader.conn.execute("INSERT OR REPLACE INTO fno_bhav_copy SELECT * FROM df")
                            successful_types.append(eod_type)
                            logger.info(f"✅ {eod_type}: {len(df)} records (INSERT OR REPLACE)")
                    
                    elif eod_type == "equity_bhav_copy_delivery":
                        df = self.eod_downloader.nse.bhav_copy_with_delivery(nse_date_format)
                        if not df.empty:
                            df['TRADE_DATE'] = target_dt.date()
                            df['last_updated'] = datetime.now().isoformat()
                            # FIXED: Use INSERT OR REPLACE instead of DELETE + INSERT
                            self.eod_downloader.conn.execute("INSERT OR REPLACE INTO equity_bhav_copy_delivery SELECT * FROM df")
                            successful_types.append(eod_type)
                            logger.info(f"✅ {eod_type}: {len(df)} records (INSERT OR REPLACE)")
                    
                    elif eod_type == "bhav_copy_indices":
                        df = self.eod_downloader.nse.bhav_copy_indices(nse_date_format)
                        if not df.empty:
                            df['TRADE_DATE'] = target_dt.date()
                            df['last_updated'] = datetime.now().isoformat()
                            # FIXED: Use INSERT OR REPLACE instead of DELETE + INSERT
                            self.eod_downloader.conn.execute("INSERT OR REPLACE INTO bhav_copy_indices SELECT * FROM df")
                            successful_types.append(eod_type)
                            logger.info(f"✅ {eod_type}: {len(df)} records (INSERT OR REPLACE)")
                    
                    elif eod_type == "fii_dii_activity":
                        df = self.eod_downloader.nse.fii_dii_activity()
                        if not df.empty:
                            df['activity_date'] = target_dt.date()
                            df['last_updated'] = datetime.now().isoformat()
                            # FIXED: Use INSERT OR REPLACE instead of DELETE + INSERT
                            self.eod_downloader.conn.execute("INSERT OR REPLACE INTO fii_dii_activity SELECT * FROM df")
                            successful_types.append(eod_type)
                            logger.info(f"✅ {eod_type}: {len(df)} records (INSERT OR REPLACE)")
                    
                except Exception as e:
                    failed_types.append(eod_type)
                    logger.error(f"❌ Failed to download {eod_type}: {e}")
            
            return {
                "success": True,
                "successful_types": successful_types,
                "failed_types": failed_types,
                "total_types": len(eod_types)
            }
            
        except Exception as e:
            logger.error(f"EOD extra data update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_all_tickers(self, tickers: List[str], target_date: str) -> Dict[str, Any]:
        """Update all tickers for the target date."""
        try:
            results = []
            successful = 0
            failed = 0
            
            for ticker in tickers:
                result = await self._update_single_ticker(ticker, target_date)
                results.append(result)
                
                if result.get("success", False):
                    successful += 1
                else:
                    failed += 1
            
            return {
                "success": True,
                "successful": successful,
                "failed": failed,
                "total": len(tickers),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to update all tickers: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_single_ticker(self, ticker: str, target_date: str) -> Dict[str, Any]:
        """Update a single ticker for the target date."""
        try:
            # Check update requirements
            requirements = await self._check_update_requirements(ticker, target_date)
            
            if not requirements['needs_update']:
                return {"success": True, "ticker": ticker, "skipped": True}
            
            # Collect data
            config = DataCollectionConfig(
                ticker=ticker,
                start_date=requirements['start_date'],
                end_date=target_date,
                data_types=self.tickers_config.get("data_types", ["technical"])
            )
            
            results = await self.collector.collect_data(config)
            
            # Store collected data
            total_records = await self._store_collected_data(ticker, results)
            
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
                    ticker, "technical", target_date, target_date
                )
                missing_data_counts.append(len(missing_dates))
            
            return {
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "total_missing_data_points": sum(missing_data_counts),
                "tickers_with_missing_data": sum(1 for count in missing_data_counts if count > 0)
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
        """Get current update status and statistics."""
        try:
            all_tickers = (self.tickers_config.get("indian_stocks", []) + 
                          self.tickers_config.get("us_stocks", []))
            
            status = {
                "total_tickers": len(all_tickers),
                "indian_stocks": len(self.tickers_config.get("indian_stocks", [])),
                "us_stocks": len(self.tickers_config.get("us_stocks", [])),
                "data_types": self.tickers_config.get("data_types", []),
                "eod_extra_data_enabled": self.tickers_config.get("eod_extra_data", {}).get("enabled", False),
                "eod_extra_data_types": self.tickers_config.get("eod_extra_data", {}).get("types", [])
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get update status: {e}")
            return {"error": str(e)}