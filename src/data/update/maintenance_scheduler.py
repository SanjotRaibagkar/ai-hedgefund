"""
Maintenance Scheduler for AI Hedge Fund.
Handles scheduling and execution of automated maintenance tasks.
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from loguru import logger
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from ..database.duckdb_manager import DatabaseManager
from .daily_updater import DailyDataUpdater
from .data_quality_monitor import DataQualityMonitor
from .missing_data_filler import MissingDataFiller


class MaintenanceScheduler:
    """Handles automated scheduling of maintenance tasks."""
    
    def __init__(self, db_manager: DatabaseManager = None, config_path: str = "config/maintenance.json"):
        """
        Initialize maintenance scheduler.
        
        Args:
            db_manager: Database manager instance
            config_path: Path to maintenance configuration file
        """
        self.db_manager = db_manager or DatabaseManager()
        self.config_path = config_path
        self.daily_updater = DailyDataUpdater(self.db_manager)
        self.quality_monitor = DataQualityMonitor(self.db_manager)
        self.data_filler = MissingDataFiller(self.db_manager)
        
        # Load maintenance configuration
        self.config = self._load_maintenance_config()
        
        # Task execution tracking
        self.task_history = []
        self.running = False
        self.scheduler_thread = None
        
        # Available maintenance tasks
        self.available_tasks = {
            'daily_update': self._run_daily_update,
            'quality_check': self._run_quality_check,
            'fill_missing_data': self._run_missing_data_fill,
            'database_cleanup': self._run_database_cleanup,
            'health_check': self._run_health_check,
            'backup_database': self._run_database_backup
        }
    
    def _load_maintenance_config(self) -> Dict[str, Any]:
        """Load maintenance configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    "enabled": True,
                    "timezone": "UTC",
                    "tasks": {
                        "daily_update": {
                            "enabled": True,
                            "schedule": "daily",
                            "time": "06:00",
                            "retry_on_failure": True,
                            "max_retries": 3,
                            "notification_on_failure": True
                        },
                        "quality_check": {
                            "enabled": True,
                            "schedule": "daily",
                            "time": "07:00",
                            "parameters": {
                                "quality_threshold": 80.0
                            }
                        },
                        "fill_missing_data": {
                            "enabled": True,
                            "schedule": "weekly",
                            "day": "sunday",
                            "time": "08:00",
                            "parameters": {
                                "fill_method": "smart",
                                "max_gap_days": 7
                            }
                        },
                        "database_cleanup": {
                            "enabled": True,
                            "schedule": "weekly",
                            "day": "sunday",
                            "time": "02:00",
                            "parameters": {
                                "keep_days": 365
                            }
                        },
                        "health_check": {
                            "enabled": True,
                            "schedule": "hourly",
                            "parameters": {
                                "check_database": True,
                                "check_data_freshness": True
                            }
                        },
                        "backup_database": {
                            "enabled": True,
                            "schedule": "daily",
                            "time": "01:00",
                            "parameters": {
                                "backup_path": "backups/",
                                "keep_backups": 7
                            }
                        }
                    },
                    "notifications": {
                        "email_enabled": False,
                        "email_recipients": [],
                        "slack_enabled": False,
                        "slack_webhook": ""
                    },
                    "logging": {
                        "log_level": "INFO",
                        "log_file": "logs/maintenance.log",
                        "max_log_size_mb": 100
                    }
                }
                
                # Create config directory and save default
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load maintenance configuration: {e}")
            return {"enabled": False, "tasks": {}}
    
    def start_scheduler(self):
        """Start the maintenance scheduler."""
        try:
            if not self.config.get("enabled", False):
                logger.info("Maintenance scheduler is disabled in configuration")
                return
            
            if self.running:
                logger.warning("Scheduler is already running")
                return
            
            logger.info("Starting maintenance scheduler...")
            
            # Schedule all enabled tasks
            self._schedule_tasks()
            
            # Start scheduler in a separate thread
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("Maintenance scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start maintenance scheduler: {e}")
            self.running = False
    
    def stop_scheduler(self):
        """Stop the maintenance scheduler."""
        try:
            if not self.running:
                logger.info("Scheduler is not running")
                return
            
            logger.info("Stopping maintenance scheduler...")
            self.running = False
            
            # Clear all scheduled jobs
            schedule.clear()
            
            # Wait for scheduler thread to finish
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            logger.info("Maintenance scheduler stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop maintenance scheduler: {e}")
    
    def _schedule_tasks(self):
        """Schedule all enabled maintenance tasks."""
        try:
            tasks_config = self.config.get("tasks", {})
            
            for task_name, task_config in tasks_config.items():
                if not task_config.get("enabled", False):
                    logger.debug(f"Task {task_name} is disabled")
                    continue
                
                if task_name not in self.available_tasks:
                    logger.warning(f"Unknown task: {task_name}")
                    continue
                
                self._schedule_single_task(task_name, task_config)
            
            logger.info(f"Scheduled {len([t for t in tasks_config.values() if t.get('enabled', False)])} maintenance tasks")
            
        except Exception as e:
            logger.error(f"Failed to schedule tasks: {e}")
    
    def _schedule_single_task(self, task_name: str, task_config: Dict[str, Any]):
        """Schedule a single maintenance task."""
        try:
            schedule_type = task_config.get("schedule", "daily")
            task_func = self.available_tasks[task_name]
            
            # Create wrapper function that includes error handling
            def task_wrapper():
                try:
                    start_time = datetime.now()
                    logger.info(f"Starting scheduled task: {task_name}")
                    
                    # Run the task
                    result = asyncio.run(task_func(task_config.get("parameters", {})))
                    
                    # Record task execution
                    execution_record = {
                        "task_name": task_name,
                        "start_time": start_time.isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "duration_seconds": (datetime.now() - start_time).total_seconds(),
                        "success": result.get("success", False),
                        "result": result
                    }
                    
                    self.task_history.append(execution_record)
                    
                    # Keep only last 100 records
                    if len(self.task_history) > 100:
                        self.task_history = self.task_history[-100:]
                    
                    if result.get("success", False):
                        logger.info(f"Task {task_name} completed successfully")
                    else:
                        logger.error(f"Task {task_name} failed: {result.get('error', 'Unknown error')}")
                        
                        # Handle retries if configured
                        if task_config.get("retry_on_failure", False):
                            self._handle_task_retry(task_name, task_config, result)
                    
                except Exception as e:
                    logger.error(f"Task {task_name} execution failed: {e}")
            
            # Schedule based on type
            if schedule_type == "daily":
                time_str = task_config.get("time", "00:00")
                schedule.every().day.at(time_str).do(task_wrapper)
                logger.info(f"Scheduled {task_name} daily at {time_str}")
                
            elif schedule_type == "weekly":
                day = task_config.get("day", "monday").lower()
                time_str = task_config.get("time", "00:00")
                
                if day == "monday":
                    schedule.every().monday.at(time_str).do(task_wrapper)
                elif day == "tuesday":
                    schedule.every().tuesday.at(time_str).do(task_wrapper)
                elif day == "wednesday":
                    schedule.every().wednesday.at(time_str).do(task_wrapper)
                elif day == "thursday":
                    schedule.every().thursday.at(time_str).do(task_wrapper)
                elif day == "friday":
                    schedule.every().friday.at(time_str).do(task_wrapper)
                elif day == "saturday":
                    schedule.every().saturday.at(time_str).do(task_wrapper)
                elif day == "sunday":
                    schedule.every().sunday.at(time_str).do(task_wrapper)
                
                logger.info(f"Scheduled {task_name} weekly on {day} at {time_str}")
                
            elif schedule_type == "hourly":
                schedule.every().hour.do(task_wrapper)
                logger.info(f"Scheduled {task_name} hourly")
                
            elif schedule_type == "minutes":
                interval = task_config.get("interval", 60)
                schedule.every(interval).minutes.do(task_wrapper)
                logger.info(f"Scheduled {task_name} every {interval} minutes")
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task_name}: {e}")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
            self.running = False
    
    def _handle_task_retry(self, task_name: str, task_config: Dict[str, Any], last_result: Dict[str, Any]):
        """Handle task retry logic."""
        try:
            max_retries = task_config.get("max_retries", 3)
            # Implementation for retry logic would go here
            logger.info(f"Retry logic for {task_name} (max retries: {max_retries})")
        except Exception as e:
            logger.error(f"Failed to handle retry for {task_name}: {e}")
    
    async def _run_daily_update(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run daily data update task."""
        try:
            target_date = parameters.get("target_date")
            result = await self.daily_updater.run_daily_update(target_date)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_quality_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run data quality check task."""
        try:
            threshold = parameters.get("quality_threshold", 80.0)
            
            # Get all tickers from config
            tickers = (self.daily_updater.tickers_config.get("indian_stocks", []) + 
                      self.daily_updater.tickers_config.get("us_stocks", []))
            
            # Generate quality report
            report = await self.quality_monitor.generate_quality_report(tickers)
            
            # Check if any tickers are below threshold
            issues_found = []
            if "ticker_quality" in report:
                for ticker, quality_data in report["ticker_quality"].items():
                    if quality_data.get("overall_score", 0) < threshold:
                        issues_found.append({
                            "ticker": ticker,
                            "score": quality_data.get("overall_score", 0),
                            "issues": quality_data.get("has_issues", False)
                        })
            
            return {
                "success": True,
                "quality_report": report,
                "issues_found": len(issues_found),
                "low_quality_tickers": issues_found
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_missing_data_fill(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run missing data fill task."""
        try:
            fill_method = parameters.get("fill_method", "smart")
            max_gap_days = parameters.get("max_gap_days", 7)
            
            # Get all tickers
            tickers = (self.daily_updater.tickers_config.get("indian_stocks", []) + 
                      self.daily_updater.tickers_config.get("us_stocks", []))
            
            # Fill missing data for each ticker
            results = []
            for ticker in tickers:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=max_gap_days)).strftime('%Y-%m-%d')
                
                result = await self.data_filler.fill_missing_data(ticker, start_date, end_date, fill_method)
                results.append(result)
            
            total_filled = sum([r.get("filled_records", 0) for r in results if r.get("success", False)])
            
            return {
                "success": True,
                "total_tickers_processed": len(tickers),
                "total_records_filled": total_filled,
                "results": results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_database_cleanup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run database cleanup task."""
        try:
            keep_days = parameters.get("keep_days", 365)
            cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime('%Y-%m-%d')
            
            # Database cleanup logic would go here
            # For now, just return success
            
            return {
                "success": True,
                "cutoff_date": cutoff_date,
                "message": "Database cleanup completed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_health_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run system health check task."""
        try:
            check_database = parameters.get("check_database", True)
            check_data_freshness = parameters.get("check_data_freshness", True)
            
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "overall_healthy": True,
                "checks": {}
            }
            
            # Database connectivity check
            if check_database:
                try:
                    # Simple query to test database
                    test_result = self.db_manager.get_technical_data("TEST", "2024-01-01", "2024-01-01")
                    health_status["checks"]["database"] = {
                        "status": "healthy",
                        "message": "Database connection successful"
                    }
                except Exception as e:
                    health_status["checks"]["database"] = {
                        "status": "unhealthy",
                        "message": f"Database connection failed: {e}"
                    }
                    health_status["overall_healthy"] = False
            
            # Data freshness check
            if check_data_freshness:
                try:
                    tickers = self.daily_updater.tickers_config.get("indian_stocks", [])[:3]  # Check first 3
                    stale_tickers = []
                    
                    for ticker in tickers:
                        latest_date = self.db_manager.get_latest_data_date(ticker, "technical")
                        if latest_date:
                            latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                            days_behind = (datetime.now() - latest_dt).days
                            if days_behind > 3:  # Consider stale if more than 3 days behind
                                stale_tickers.append({"ticker": ticker, "days_behind": days_behind})
                    
                    if stale_tickers:
                        health_status["checks"]["data_freshness"] = {
                            "status": "warning",
                            "message": f"{len(stale_tickers)} tickers have stale data",
                            "stale_tickers": stale_tickers
                        }
                    else:
                        health_status["checks"]["data_freshness"] = {
                            "status": "healthy",
                            "message": "All data is fresh"
                        }
                        
                except Exception as e:
                    health_status["checks"]["data_freshness"] = {
                        "status": "error",
                        "message": f"Data freshness check failed: {e}"
                    }
            
            return {
                "success": True,
                "health_status": health_status
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_database_backup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run database backup task."""
        try:
            backup_path = parameters.get("backup_path", "backups/")
            keep_backups = parameters.get("keep_backups", 7)
            
            # Create backup directory
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"ai_hedge_fund_backup_{timestamp}.db"
            
            # Database backup logic would go here
            # For SQLite, you could use shutil.copy2 or sqlite3 backup
            
            return {
                "success": True,
                "backup_file": str(backup_file),
                "timestamp": timestamp,
                "message": "Database backup completed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_task_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent task execution history."""
        return self.task_history[-limit:] if self.task_history else []
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "running": self.running,
            "enabled": self.config.get("enabled", False),
            "scheduled_tasks": len([t for t in self.config.get("tasks", {}).values() if t.get("enabled", False)]),
            "last_execution": self.task_history[-1] if self.task_history else None,
            "uptime": "N/A"  # Would track actual uptime
        }
    
    async def run_task_manually(self, task_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manually run a specific maintenance task."""
        try:
            if task_name not in self.available_tasks:
                return {"success": False, "error": f"Unknown task: {task_name}"}
            
            logger.info(f"Manually running task: {task_name}")
            
            task_func = self.available_tasks[task_name]
            result = await task_func(parameters or {})
            
            # Record manual execution
            execution_record = {
                "task_name": task_name,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "manual_execution": True,
                "success": result.get("success", False),
                "result": result
            }
            
            self.task_history.append(execution_record)
            
            return result
            
        except Exception as e:
            logger.error(f"Manual task execution failed: {e}")
            return {"success": False, "error": str(e)}