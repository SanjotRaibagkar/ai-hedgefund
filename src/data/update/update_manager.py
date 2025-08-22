"""
Update Manager for AI Hedge Fund.
Central manager for coordinating all data update and maintenance operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from pathlib import Path
import json

from ..database.duckdb_manager import DatabaseManager
from .daily_updater import DailyDataUpdater
from .data_quality_monitor import DataQualityMonitor
from .missing_data_filler import MissingDataFiller
from .maintenance_scheduler import MaintenanceScheduler


class UpdateManager:
    """Central manager for all data update and maintenance operations."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize update manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        
        # Initialize all update components
        self.daily_updater = DailyDataUpdater(self.db_manager)
        self.quality_monitor = DataQualityMonitor(self.db_manager)
        self.data_filler = MissingDataFiller(self.db_manager)
        self.scheduler = MaintenanceScheduler(self.db_manager)
        
        # Status tracking
        self.last_update_status = None
        self.update_in_progress = False
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the update manager and all components.
        
        Returns:
            Initialization status
        """
        try:
            logger.info("Initializing Update Manager...")
            
            # Check database connectivity
            try:
                # Test database connection
                test_result = self.db_manager.get_technical_data("TEST", "2024-01-01", "2024-01-01")
                logger.info("✅ Database connection verified")
            except Exception as e:
                logger.error(f"❌ Database connection failed: {e}")
                return {"success": False, "error": "Database connection failed"}
            
            # Load configurations
            try:
                ticker_config = self.daily_updater.tickers_config
                maintenance_config = self.scheduler.config
                logger.info(f"✅ Configurations loaded - {len(ticker_config.get('indian_stocks', []) + ticker_config.get('us_stocks', []))} tickers configured")
            except Exception as e:
                logger.error(f"❌ Configuration loading failed: {e}")
                return {"success": False, "error": "Configuration loading failed"}
            
            # Initialize scheduler if enabled
            if maintenance_config.get("enabled", False):
                self.scheduler.start_scheduler()
                logger.info("✅ Maintenance scheduler started")
            else:
                logger.info("ℹ️ Maintenance scheduler disabled")
            
            initialization_result = {
                "success": True,
                "initialized_at": datetime.now().isoformat(),
                "database_status": "connected",
                "scheduler_status": "running" if maintenance_config.get("enabled", False) else "disabled",
                "configured_tickers": len(ticker_config.get('indian_stocks', []) + ticker_config.get('us_stocks', [])),
                "components": {
                    "daily_updater": "ready",
                    "quality_monitor": "ready",
                    "data_filler": "ready",
                    "scheduler": "running" if maintenance_config.get("enabled", False) else "disabled"
                }
            }
            
            logger.info("🚀 Update Manager initialized successfully")
            return initialization_result
            
        except Exception as e:
            logger.error(f"❌ Update Manager initialization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_update(self, target_date: str = None, 
                                     include_quality_check: bool = True,
                                     fill_missing_data: bool = True) -> Dict[str, Any]:
        """
        Run a comprehensive update including data collection, quality checks, and gap filling.
        
        Args:
            target_date: Target date for update (YYYY-MM-DD)
            include_quality_check: Whether to run quality checks
            fill_missing_data: Whether to fill missing data
            
        Returns:
            Comprehensive update results
        """
        try:
            if self.update_in_progress:
                return {"success": False, "error": "Update already in progress"}
            
            self.update_in_progress = True
            start_time = datetime.now()
            
            logger.info("🔄 Starting comprehensive data update...")
            
            comprehensive_result = {
                "success": True,
                "started_at": start_time.isoformat(),
                "target_date": target_date or (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                "phases": {}
            }
            
            # Phase 1: Daily Data Update
            logger.info("📊 Phase 1: Running daily data update...")
            daily_result = await self.daily_updater.run_daily_update(target_date)
            comprehensive_result["phases"]["daily_update"] = daily_result
            
            if not daily_result.get("success", False):
                logger.error("❌ Daily update failed, stopping comprehensive update")
                comprehensive_result["success"] = False
                return comprehensive_result
            
            # Phase 2: Quality Check (if enabled)
            if include_quality_check:
                logger.info("🔍 Phase 2: Running quality checks...")
                tickers = (self.daily_updater.tickers_config.get("indian_stocks", []) + 
                          self.daily_updater.tickers_config.get("us_stocks", []))
                
                quality_result = await self.quality_monitor.generate_quality_report(tickers)
                comprehensive_result["phases"]["quality_check"] = quality_result
            
            # Phase 3: Fill Missing Data (if enabled)
            if fill_missing_data:
                logger.info("🔧 Phase 3: Filling missing data...")
                
                tickers = (self.daily_updater.tickers_config.get("indian_stocks", []) + 
                          self.daily_updater.tickers_config.get("us_stocks", []))
                
                missing_data_results = []
                target_date_str = comprehensive_result["target_date"]
                start_date = (datetime.strptime(target_date_str, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
                
                for ticker in tickers[:5]:  # Limit to first 5 tickers for efficiency
                    fill_result = await self.data_filler.fill_missing_data(ticker, start_date, target_date_str, "smart")
                    missing_data_results.append(fill_result)
                
                comprehensive_result["phases"]["missing_data_fill"] = {
                    "success": True,
                    "tickers_processed": len(missing_data_results),
                    "total_records_filled": sum([r.get("filled_records", 0) for r in missing_data_results if r.get("success", False)]),
                    "results": missing_data_results
                }
            
            # Calculate overall duration and success
            end_time = datetime.now()
            comprehensive_result["completed_at"] = end_time.isoformat()
            comprehensive_result["total_duration_seconds"] = (end_time - start_time).total_seconds()
            
            # Store last update status
            self.last_update_status = comprehensive_result
            
            logger.info(f"✅ Comprehensive update completed in {comprehensive_result['total_duration_seconds']:.1f} seconds")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ Comprehensive update failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.update_in_progress = False
    
    async def quick_update(self, tickers: List[str] = None, target_date: str = None) -> Dict[str, Any]:
        """
        Run a quick update for specific tickers or recent data.
        
        Args:
            tickers: List of specific tickers to update
            target_date: Target date for update
            
        Returns:
            Quick update results
        """
        try:
            logger.info("⚡ Starting quick update...")
            
            if not tickers:
                # Use first 3 Indian stocks for quick update
                tickers = self.daily_updater.tickers_config.get("indian_stocks", [])[:3]
            
            if not target_date:
                target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            quick_results = []
            
            for ticker in tickers:
                result = await self.daily_updater._update_single_ticker(ticker, target_date)
                quick_results.append(result)
            
            successful = len([r for r in quick_results if r.get("success", False)])
            total_records = sum([r.get("records_collected", 0) for r in quick_results if r.get("success", False)])
            
            return {
                "success": True,
                "type": "quick_update",
                "target_date": target_date,
                "tickers_processed": len(tickers),
                "successful_updates": successful,
                "total_records_collected": total_records,
                "results": quick_results,
                "completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Quick update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def emergency_data_recovery(self, ticker: str, start_date: str, 
                                    end_date: str) -> Dict[str, Any]:
        """
        Emergency data recovery for a specific ticker and date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for recovery (YYYY-MM-DD)
            end_date: End date for recovery (YYYY-MM-DD)
            
        Returns:
            Recovery results
        """
        try:
            logger.info(f"🚨 Emergency data recovery for {ticker} from {start_date} to {end_date}")
            
            # Try multiple recovery methods
            recovery_results = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "recovery_methods": {},
                "total_recovered": 0
            }
            
            # Method 1: API retry
            logger.info("🔄 Attempting API retry recovery...")
            api_result = await self.data_filler.fill_missing_data(ticker, start_date, end_date, "api_retry")
            recovery_results["recovery_methods"]["api_retry"] = api_result
            
            if api_result.get("success", False):
                recovery_results["total_recovered"] += api_result.get("filled_records", 0)
            
            # Method 2: Interpolation for remaining gaps
            logger.info("🔧 Attempting interpolation recovery...")
            interp_result = await self.data_filler.fill_missing_data(ticker, start_date, end_date, "interpolation")
            recovery_results["recovery_methods"]["interpolation"] = interp_result
            
            if interp_result.get("success", False):
                recovery_results["total_recovered"] += interp_result.get("filled_records", 0)
            
            # Method 3: Hybrid approach as last resort
            if recovery_results["total_recovered"] == 0:
                logger.info("🔀 Attempting hybrid recovery...")
                hybrid_result = await self.data_filler.fill_missing_data(ticker, start_date, end_date, "hybrid")
                recovery_results["recovery_methods"]["hybrid"] = hybrid_result
                
                if hybrid_result.get("success", False):
                    recovery_results["total_recovered"] += hybrid_result.get("filled_records", 0)
            
            # Update quality metrics after recovery
            await self.quality_monitor.update_quality_metrics(ticker, end_date)
            
            recovery_results["success"] = recovery_results["total_recovered"] > 0
            recovery_results["completed_at"] = datetime.now().isoformat()
            
            if recovery_results["success"]:
                logger.info(f"✅ Emergency recovery completed: {recovery_results['total_recovered']} records recovered")
            else:
                logger.warning(f"⚠️ Emergency recovery found no data for {ticker}")
            
            return recovery_results
            
        except Exception as e:
            logger.error(f"❌ Emergency data recovery failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get update status
            update_status = self.daily_updater.get_update_status()
            
            # Get scheduler status
            scheduler_status = self.scheduler.get_scheduler_status()
            
            # Get recent task history
            recent_tasks = self.scheduler.get_task_history(10)
            
            # Calculate system health
            healthy_tickers = 0
            total_tickers = update_status.get("total_tickers", 0)
            
            for ticker_status in update_status.get("ticker_status", {}).values():
                if not ticker_status.get("needs_update", True):
                    healthy_tickers += 1
            
            health_score = (healthy_tickers / total_tickers * 100) if total_tickers > 0 else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_health": {
                    "overall_score": health_score,
                    "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
                    "healthy_tickers": healthy_tickers,
                    "total_tickers": total_tickers
                },
                "update_status": update_status,
                "scheduler_status": scheduler_status,
                "last_comprehensive_update": self.last_update_status,
                "update_in_progress": self.update_in_progress,
                "recent_maintenance_tasks": recent_tasks
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def run_health_diagnosis(self) -> Dict[str, Any]:
        """Run comprehensive health diagnosis of the system."""
        try:
            logger.info("🏥 Running system health diagnosis...")
            
            diagnosis_result = {
                "timestamp": datetime.now().isoformat(),
                "overall_health": "unknown",
                "issues": [],
                "recommendations": [],
                "checks": {}
            }
            
            # Check 1: Database connectivity
            try:
                test_result = self.db_manager.get_technical_data("TEST", "2024-01-01", "2024-01-01")
                diagnosis_result["checks"]["database"] = {"status": "healthy", "message": "Database accessible"}
            except Exception as e:
                diagnosis_result["checks"]["database"] = {"status": "error", "message": f"Database error: {e}"}
                diagnosis_result["issues"].append("Database connectivity issue")
                diagnosis_result["recommendations"].append("Check database connection and credentials")
            
            # Check 2: Data freshness
            tickers = self.daily_updater.tickers_config.get("indian_stocks", [])[:5]  # Check first 5
            stale_data_count = 0
            
            for ticker in tickers:
                latest_date = self.db_manager.get_latest_data_date(ticker, "technical")
                if latest_date:
                    latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                    days_behind = (datetime.now() - latest_dt).days
                    if days_behind > 3:
                        stale_data_count += 1
            
            if stale_data_count > 0:
                diagnosis_result["checks"]["data_freshness"] = {
                    "status": "warning",
                    "message": f"{stale_data_count}/{len(tickers)} tickers have stale data"
                }
                diagnosis_result["issues"].append(f"Stale data detected for {stale_data_count} tickers")
                diagnosis_result["recommendations"].append("Run comprehensive update to refresh stale data")
            else:
                diagnosis_result["checks"]["data_freshness"] = {
                    "status": "healthy",
                    "message": "All checked tickers have fresh data"
                }
            
            # Check 3: System configuration
            config_issues = []
            if not self.daily_updater.tickers_config.get("indian_stocks"):
                config_issues.append("No Indian stocks configured")
            if not self.daily_updater.tickers_config.get("us_stocks"):
                config_issues.append("No US stocks configured")
            
            if config_issues:
                diagnosis_result["checks"]["configuration"] = {
                    "status": "warning",
                    "message": f"Configuration issues: {', '.join(config_issues)}"
                }
                diagnosis_result["issues"].extend(config_issues)
                diagnosis_result["recommendations"].append("Review and update ticker configuration")
            else:
                diagnosis_result["checks"]["configuration"] = {
                    "status": "healthy",
                    "message": "Configuration appears valid"
                }
            
            # Determine overall health
            error_count = len([c for c in diagnosis_result["checks"].values() if c["status"] == "error"])
            warning_count = len([c for c in diagnosis_result["checks"].values() if c["status"] == "warning"])
            
            if error_count > 0:
                diagnosis_result["overall_health"] = "critical"
            elif warning_count > 0:
                diagnosis_result["overall_health"] = "warning"
            else:
                diagnosis_result["overall_health"] = "healthy"
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"Health diagnosis failed: {e}")
            return {"overall_health": "error", "error": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the update manager."""
        try:
            logger.info("🔄 Shutting down Update Manager...")
            
            # Stop scheduler
            self.scheduler.stop_scheduler()
            
            # Close database connections
            self.db_manager.close()
            
            logger.info("✅ Update Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Update Manager shutdown failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.shutdown())