"""
Maintenance Scheduler for AI Hedge Fund.
Handles automated scheduling of data updates and maintenance tasks.
"""

import asyncio
import schedule
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger
import json
import os
from pathlib import Path

from .simple_price_updater import SimplePriceUpdater
from ..database.duckdb_manager import DatabaseManager
from ..downloaders.eod_extra_data_downloader import EODExtraDataDownloader


class MaintenanceScheduler:
    """Handles automated scheduling of data updates and maintenance tasks."""
    
    def __init__(self, config_path: str = "config/scheduler_config.json"):
        """
        Initialize maintenance scheduler.
        
        Args:
            config_path: Path to scheduler configuration file
        """
        self.config_path = config_path
        self.db_manager = DatabaseManager()
        self.daily_updater = SimplePriceUpdater()
        self.eod_downloader = EODExtraDataDownloader()
        
        # Load scheduler configuration
        self.scheduler_config = self._load_scheduler_config()
        
        # Initialize schedule
        self._setup_schedules()
        
        logger.info("Maintenance Scheduler initialized")
    
    def _load_scheduler_config(self) -> Dict[str, Any]:
        """Load scheduler configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    "daily_updates": {
                        "enabled": True,
                        "time": "06:00",  # 6 AM
                        "timezone": "Asia/Kolkata",
                        "include_eod_extra_data": True,
                        "retry_attempts": 3,
                        "retry_delay_minutes": 30
                    },
                    "weekly_maintenance": {
                        "enabled": True,
                        "day": "sunday",
                        "time": "02:00",
                        "timezone": "Asia/Kolkata"
                    },
                    "monthly_cleanup": {
                        "enabled": True,
                        "day": 1,
                        "time": "03:00",
                        "timezone": "Asia/Kolkata"
                    },
                    "eod_extra_data": {
                        "enabled": True,
                        "time": "06:00",  # 6 AM as requested
                        "timezone": "Asia/Kolkata",
                        "data_types": ["fno_bhav_copy", "equity_bhav_copy_delivery", "bhav_copy_indices", "fii_dii_activity"],
                        "retry_attempts": 3,
                        "retry_delay_minutes": 15
                    },
                    "notifications": {
                        "enabled": True,
                        "email": False,
                        "log_file": True,
                        "console": True
                    },
                    "performance": {
                        "max_concurrent_jobs": 3,
                        "job_timeout_minutes": 120,
                        "memory_limit_mb": 2048
                    }
                }
                
                # Create config directory and save default
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load scheduler configuration: {e}")
            return {
                "daily_updates": {"enabled": True, "time": "06:00"},
                "eod_extra_data": {"enabled": True, "time": "06:00"},
                "notifications": {"enabled": True}
            }
    
    def _setup_schedules(self):
        """Setup all scheduled jobs."""
        try:
            # Clear existing schedules
            schedule.clear()
            
            # Daily updates at 6 AM
            if self.scheduler_config.get("daily_updates", {}).get("enabled", True):
                daily_time = self.scheduler_config["daily_updates"].get("time", "06:00")
                schedule.every().day.at(daily_time).do(self._run_daily_update)
                logger.info(f"✅ Scheduled daily updates at {daily_time}")
            
            # EOD Extra Data updates at 6 AM
            if self.scheduler_config.get("eod_extra_data", {}).get("enabled", True):
                eod_time = self.scheduler_config["eod_extra_data"].get("time", "06:00")
                schedule.every().day.at(eod_time).do(self._run_eod_extra_data_update)
                logger.info(f"✅ Scheduled EOD extra data updates at {eod_time}")
            
            # Weekly maintenance
            if self.scheduler_config.get("weekly_maintenance", {}).get("enabled", True):
                weekly_day = self.scheduler_config["weekly_maintenance"].get("day", "sunday")
                weekly_time = self.scheduler_config["weekly_maintenance"].get("time", "02:00")
                getattr(schedule.every(), weekly_day).at(weekly_time).do(self._run_weekly_maintenance)
                logger.info(f"✅ Scheduled weekly maintenance on {weekly_day} at {weekly_time}")
            
            # Monthly cleanup (using weekly schedule as fallback since schedule library doesn't support monthly)
            if self.scheduler_config.get("monthly_cleanup", {}).get("enabled", True):
                monthly_time = self.scheduler_config["monthly_cleanup"].get("time", "03:00")
                # Schedule for first Sunday of each month (approximation)
                schedule.every().sunday.at(monthly_time).do(self._run_monthly_cleanup)
                logger.info(f"✅ Scheduled monthly cleanup (weekly fallback) at {monthly_time}")
            
            logger.info("All schedules configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup schedules: {e}")
    
    def _run_daily_update(self):
        """Run daily data update."""
        try:
            logger.info("🔄 Starting scheduled daily update")
            
            # Get yesterday's date
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Run daily update
            self.daily_updater.run_daily_update(yesterday)
            
            logger.info("✅ Daily update completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Daily update failed: {e}")
            self._handle_job_failure("daily_update", str(e))
    
    def _run_eod_extra_data_update(self):
        """Run EOD extra data update."""
        try:
            logger.info("🔄 Starting scheduled EOD extra data update")
            
            # Get yesterday's date
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Run EOD extra data update
            asyncio.run(self._update_eod_extra_data(yesterday))
            
            logger.info("✅ EOD extra data update completed successfully")
            
        except Exception as e:
            logger.error(f"❌ EOD extra data update failed: {e}")
            self._handle_job_failure("eod_extra_data_update", str(e))
    
    async def _update_eod_extra_data(self, target_date: str):
        """Update EOD extra data for the target date."""
        try:
            eod_types = self.scheduler_config.get("eod_extra_data", {}).get("data_types", [])
            successful_types = []
            failed_types = []
            
            # Convert target_date to DD-MM-YYYY format for NSE API
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            nse_date_format = target_dt.strftime('%d-%m-%Y')
            
            logger.info(f"Downloading EOD extra data for {nse_date_format}")
            
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
                            # Clean problematic data - handle '-' values
                            if 'DELIV_QTY' in df.columns:
                                df['DELIV_QTY'] = df['DELIV_QTY'].replace(['-', ' -', '- '], '0')
                                df['DELIV_QTY'] = pd.to_numeric(df['DELIV_QTY'], errors='coerce').fillna(0).astype(int)
                            if 'DELIV_PER' in df.columns:
                                df['DELIV_PER'] = df['DELIV_PER'].replace(['-', ' -', '- '], '0')
                                df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'], errors='coerce').fillna(0)
                            
                            df['TRADE_DATE'] = target_dt.date()
                            df['last_updated'] = datetime.now().isoformat()
                            # FIXED: Use INSERT OR REPLACE instead of DELETE + INSERT
                            # This preserves existing data and only updates duplicates
                            self.eod_downloader.conn.execute("INSERT OR REPLACE INTO equity_bhav_copy_delivery SELECT * FROM df")
                            successful_types.append(eod_type)
                            logger.info(f"✅ {eod_type}: {len(df)} records (INSERT OR REPLACE)")
                    
                    elif eod_type == "bhav_copy_indices":
                        df = self.eod_downloader.nse.bhav_copy_indices(nse_date_format)
                        if not df.empty:
                            # Clean problematic data - handle '-' values in numeric columns
                            numeric_columns = ['Open Index Value', 'High Index Value', 'Low Index Value', 
                                             'Closing Index Value', 'Points Change', 'Change(%)', 'Volume', 
                                             'Turnover (Rs. Cr.)', 'P/E', 'P/B', 'Div Yield']
                            
                            for col in numeric_columns:
                                if col in df.columns:
                                    df[col] = df[col].replace(['-', ' -', '- '], '0')
                                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                            
                            df['TRADE_DATE'] = target_dt.date()
                            df['last_updated'] = datetime.now().isoformat()
                            # FIXED: Use INSERT OR REPLACE instead of DELETE + INSERT
                            # This preserves existing data and only updates duplicates
                            self.eod_downloader.conn.execute("INSERT OR REPLACE INTO bhav_copy_indices SELECT * FROM df")
                            successful_types.append(eod_type)
                            logger.info(f"✅ {eod_type}: {len(df)} records (INSERT OR REPLACE)")
                    
                    elif eod_type == "fii_dii_activity":
                        df = self.eod_downloader.nse.fii_dii_activity()
                        if not df.empty:
                            # Use 'date' column instead of 'activity_date' to match actual schema
                            df['date'] = target_dt.date()
                            df['last_updated'] = datetime.now().isoformat()
                            # FIXED: Use INSERT OR REPLACE instead of DELETE + INSERT
                            # This preserves existing data and only updates duplicates
                            self.eod_downloader.conn.execute("INSERT OR REPLACE INTO fii_dii_activity SELECT * FROM df")
                            successful_types.append(eod_type)
                            logger.info(f"✅ {eod_type}: {len(df)} records (INSERT OR REPLACE)")
                    
                except Exception as e:
                    failed_types.append(eod_type)
                    logger.error(f"❌ Failed to download {eod_type}: {e}")
            
            logger.info(f"EOD extra data update summary: {len(successful_types)} successful, {len(failed_types)} failed")
            
            return {
                "success": True,
                "successful_types": successful_types,
                "failed_types": failed_types,
                "total_types": len(eod_types)
            }
            
        except Exception as e:
            logger.error(f"EOD extra data update failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_weekly_maintenance(self):
        """Run weekly maintenance tasks."""
        try:
            logger.info("🔄 Starting weekly maintenance")
            
            # Database optimization
            self._optimize_database()
            
            # Clean up old logs
            self._cleanup_old_logs()
            
            # Generate weekly report
            self._generate_weekly_report()
            
            logger.info("✅ Weekly maintenance completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Weekly maintenance failed: {e}")
            self._handle_job_failure("weekly_maintenance", str(e))
    
    def _run_monthly_cleanup(self):
        """Run monthly cleanup tasks."""
        try:
            logger.info("🔄 Starting monthly cleanup")
            
            # Archive old data
            self._archive_old_data()
            
            # Update statistics
            self._update_monthly_statistics()
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            logger.info("✅ Monthly cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Monthly cleanup failed: {e}")
            self._handle_job_failure("monthly_cleanup", str(e))
    
    def _optimize_database(self):
        """Optimize database performance."""
        try:
            logger.info("Optimizing database...")
            
            # Run VACUUM to reclaim space
            self.db_manager.execute_query("VACUUM")
            
            # Analyze tables for better query planning
            self.db_manager.execute_query("ANALYZE")
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    def _cleanup_old_logs(self):
        """Clean up old log files."""
        try:
            logger.info("Cleaning up old logs...")
            
            # Keep logs for 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            log_dir = Path("logs")
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        logger.info(f"Deleted old log file: {log_file}")
            
            logger.info("Log cleanup completed")
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
    
    def _generate_weekly_report(self):
        """Generate weekly performance report."""
        try:
            logger.info("Generating weekly report...")
            
            # Get database statistics
            stats = self.eod_downloader.get_database_stats()
            
            # Create report
            report = {
                "generated_at": datetime.now().isoformat(),
                "period": "weekly",
                "database_stats": stats,
                "scheduler_status": self.get_scheduler_status()
            }
            
            # Save report
            report_file = Path(f"reports/weekly_report_{datetime.now().strftime('%Y%m%d')}.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Weekly report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}")
    
    def _archive_old_data(self):
        """Archive old data to save space."""
        try:
            logger.info("Archiving old data...")
            
            # Archive data older than 2 years
            cutoff_date = datetime.now() - timedelta(days=730)
            
            # Archive old price data
            self.db_manager.execute_query("""
                DELETE FROM price_data 
                WHERE date < ?
            """, [cutoff_date.date()])
            
            logger.info("Data archiving completed")
            
        except Exception as e:
            logger.error(f"Data archiving failed: {e}")
    
    def _update_monthly_statistics(self):
        """Update monthly statistics."""
        try:
            logger.info("Updating monthly statistics...")
            
            # Calculate monthly statistics
            stats = self.eod_downloader.get_database_stats()
            
            # Save monthly stats
            stats_file = Path(f"reports/monthly_stats_{datetime.now().strftime('%Y%m')}.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Monthly statistics saved: {stats_file}")
            
        except Exception as e:
            logger.error(f"Monthly statistics update failed: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            logger.info("Cleaning up temporary files...")
            
            # Clean up temporary files
            temp_dir = Path("temp")
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*"):
                    if temp_file.is_file():
                        temp_file.unlink()
                        logger.info(f"Deleted temp file: {temp_file}")
            
            logger.info("Temporary file cleanup completed")
            
        except Exception as e:
            logger.error(f"Temporary file cleanup failed: {e}")
    
    def _handle_job_failure(self, job_name: str, error: str):
        """Handle job failures with retry logic."""
        try:
            logger.error(f"Job failure: {job_name} - {error}")
            
            # Get retry configuration
            retry_attempts = self.scheduler_config.get("daily_updates", {}).get("retry_attempts", 3)
            retry_delay = self.scheduler_config.get("daily_updates", {}).get("retry_delay_minutes", 30)
            
            # Log failure
            failure_log = {
                "job_name": job_name,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "retry_attempts": retry_attempts
            }
            
            # Save failure log
            failure_file = Path(f"logs/failures/{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            failure_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(failure_file, 'w') as f:
                json.dump(failure_log, f, indent=2)
            
            logger.info(f"Failure logged: {failure_file}")
            
        except Exception as e:
            logger.error(f"Failed to handle job failure: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        try:
            return {
                "scheduler_running": True,
                "next_jobs": [],
                "last_run": {},
                "configuration": self.scheduler_config,
                "database_stats": self.eod_downloader.get_database_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}")
            return {"error": str(e)}
    
    def start(self):
        """Start the scheduler."""
        try:
            logger.info("🚀 Starting Maintenance Scheduler")
            logger.info("Press Ctrl+C to stop")
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Maintenance Scheduler stopped by user")
        except Exception as e:
            logger.error(f"❌ Scheduler failed: {e}")
    
    def run_once(self, job_type: str = "daily"):
        """Run a specific job once."""
        try:
            logger.info(f"Running {job_type} job once")
            
            if job_type == "daily":
                self._run_daily_update()
            elif job_type == "eod_extra_data":
                self._run_eod_extra_data_update()
            elif job_type == "weekly":
                self._run_weekly_maintenance()
            elif job_type == "monthly":
                self._run_monthly_cleanup()
            else:
                logger.error(f"Unknown job type: {job_type}")
                
        except Exception as e:
            logger.error(f"Failed to run {job_type} job: {e}")


def main():
    """Main function to run the scheduler."""
    print("Maintenance Scheduler for AI Hedge Fund")
    print("=" * 50)
    
    # Initialize scheduler
    scheduler = MaintenanceScheduler()
    
    # Show current status
    status = scheduler.get_scheduler_status()
    print(f"Scheduler Status: {status.get('scheduler_running', False)}")
    print(f"Configuration: {status.get('configuration', {})}")
    
    # Start scheduler
    scheduler.start()


if __name__ == "__main__":
    main()