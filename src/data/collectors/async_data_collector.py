"""
Async Data Collector for AI Hedge Fund
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from loguru import logger

from ..database.models import (
    DataCollectionConfig,
    DataCollectionResult,
    TechnicalData,
    FundamentalData,
    MarketData
)
from ..database.duckdb_manager import DuckDBManager
from .technical_collector import TechnicalDataCollector
from .fundamental_collector import FundamentalDataCollector
from .market_collector import MarketDataCollector

class AsyncDataCollector:
    """Async data collector for historical market data."""
    
    def __init__(self, db_manager: DuckDBManager, max_workers: int = 5):
        """
        Initialize async data collector.
        
        Args:
            db_manager: DuckDB manager instance
            max_workers: Maximum number of parallel workers
        """
        self.db_manager = db_manager
        self.max_workers = max_workers
        self.technical_collector = TechnicalDataCollector()
        self.fundamental_collector = FundamentalDataCollector()
        self.market_collector = MarketDataCollector()
        
    async def collect_data(self, config: DataCollectionConfig) -> List[DataCollectionResult]:
        """
        Collect data for a ticker (alias for collect_historical_data for compatibility).
        
        Args:
            config: Data collection configuration
            
        Returns:
            List of collection results
        """
        return await self.collect_historical_data(config)
    
    async def collect_historical_data(self, config: DataCollectionConfig) -> List[DataCollectionResult]:
        """
        Collect historical data for a ticker.
        
        Args:
            config: Data collection configuration
            
        Returns:
            List of collection results
        """
        logger.info(f"Starting historical data collection for {config.ticker}")
        start_time = time.time()
        
        results = []
        
        if config.parallel:
            # Collect data in parallel
            tasks = []
            for data_type in config.data_types:
                task = self._collect_data_type_async(config, data_type)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to results
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Data collection failed: {result}")
                    # Create error result
                    error_result = DataCollectionResult(
                        ticker=config.ticker,
                        success=False,
                        data_type="unknown",
                        start_date=config.start_date,
                        end_date=config.end_date,
                        duration_seconds=time.time() - start_time,
                        errors=[str(result)]
                    )
                    filtered_results.append(error_result)
                else:
                    filtered_results.append(result)
            
            results = filtered_results
        else:
            # Collect data sequentially
            for data_type in config.data_types:
                result = await self._collect_data_type_async(config, data_type)
                results.append(result)
        
        total_duration = time.time() - start_time
        logger.info(f"Historical data collection completed for {config.ticker} in {total_duration:.2f}s")
        
        return results
    
    async def _collect_data_type_async(self, config: DataCollectionConfig, data_type: str) -> DataCollectionResult:
        """
        Collect data for a specific type asynchronously.
        
        Args:
            config: Data collection configuration
            data_type: Type of data to collect
            
        Returns:
            Collection result
        """
        start_time = time.time()
        
        try:
            if data_type == 'technical':
                result = await self._collect_technical_data_async(config)
            elif data_type == 'fundamental':
                result = await self._collect_fundamental_data_async(config)
            elif data_type == 'market':
                result = await self._collect_market_data_async(config)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
            result.duration_seconds = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Failed to collect {data_type} data for {config.ticker}: {e}")
            return DataCollectionResult(
                ticker=config.ticker,
                success=False,
                data_type=data_type,
                start_date=config.start_date,
                end_date=config.end_date,
                duration_seconds=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _collect_technical_data_async(self, config: DataCollectionConfig) -> DataCollectionResult:
        """Collect technical data asynchronously."""
        try:
            # Use ThreadPoolExecutor for CPU-bound operations
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Collect data in chunks to avoid overwhelming the API
                chunk_size = 252  # One trading year
                chunks = self._create_date_chunks(config.start_date, config.end_date, chunk_size)
                
                all_data = []
                for chunk_start, chunk_end in chunks:
                    future = loop.run_in_executor(
                        executor,
                        self.technical_collector.collect_data,
                        config.ticker,
                        chunk_start.strftime('%Y-%m-%d'),
                        chunk_end.strftime('%Y-%m-%d')
                    )
                    chunk_data = await future
                    if chunk_data is not None and not chunk_data.empty:
                        all_data.append(chunk_data)
                
                if all_data:
                    # Combine all chunks
                    combined_data = pd.concat(all_data, ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=['date'])
                    combined_data = combined_data.sort_values('date')
                    
                    # Store in database
                    self.db_manager.insert_technical_data(combined_data)
                    
                    return DataCollectionResult(
                        ticker=config.ticker,
                        success=True,
                        data_type='technical',
                        records_collected=len(combined_data),
                        records_expected=self._estimate_trading_days(config.start_date, config.end_date),
                        start_date=config.start_date,
                        end_date=config.end_date,
                        duration_seconds=0  # Will be set by caller
                    )
                else:
                    return DataCollectionResult(
                        ticker=config.ticker,
                        success=False,
                        data_type='technical',
                        start_date=config.start_date,
                        end_date=config.end_date,
                        duration_seconds=0,
                        errors=["No data collected"]
                    )
                    
        except Exception as e:
            logger.error(f"Technical data collection failed for {config.ticker}: {e}")
            raise
    
    async def _collect_fundamental_data_async(self, config: DataCollectionConfig) -> DataCollectionResult:
        """Collect fundamental data asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Collect annual and quarterly data
                annual_future = loop.run_in_executor(
                    executor,
                    self.fundamental_collector.collect_annual_data,
                    config.ticker,
                    config.start_date.strftime('%Y-%m-%d'),
                    config.end_date.strftime('%Y-%m-%d')
                )
                
                quarterly_future = loop.run_in_executor(
                    executor,
                    self.fundamental_collector.collect_quarterly_data,
                    config.ticker,
                    config.start_date.strftime('%Y-%m-%d'),
                    config.end_date.strftime('%Y-%m-%d')
                )
                
                annual_data, quarterly_data = await asyncio.gather(annual_future, quarterly_future)
                
                total_records = 0
                if annual_data is not None and not annual_data.empty:
                    self.db_manager.insert_fundamental_data(annual_data)
                    total_records += len(annual_data)
                
                if quarterly_data is not None and not quarterly_data.empty:
                    self.db_manager.insert_fundamental_data(quarterly_data)
                    total_records += len(quarterly_data)
                
                return DataCollectionResult(
                    ticker=config.ticker,
                    success=True,
                    data_type='fundamental',
                    records_collected=total_records,
                    records_expected=self._estimate_fundamental_periods(config.start_date, config.end_date),
                    start_date=config.start_date,
                    end_date=config.end_date,
                    duration_seconds=0
                )
                
        except Exception as e:
            logger.error(f"Fundamental data collection failed for {config.ticker}: {e}")
            raise
    
    async def _collect_market_data_async(self, config: DataCollectionConfig) -> DataCollectionResult:
        """Collect market data asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future = loop.run_in_executor(
                    executor,
                    self.market_collector.collect_data,
                    config.ticker,
                    config.start_date.strftime('%Y-%m-%d'),
                    config.end_date.strftime('%Y-%m-%d')
                )
                
                market_data = await future
                
                if market_data is not None and not market_data.empty:
                    # Store in database (assuming we have a method for market data)
                    # self.db_manager.insert_market_data(market_data)
                    
                    return DataCollectionResult(
                        ticker=config.ticker,
                        success=True,
                        data_type='market',
                        records_collected=len(market_data),
                        records_expected=self._estimate_trading_days(config.start_date, config.end_date),
                        start_date=config.start_date,
                        end_date=config.end_date,
                        duration_seconds=0
                    )
                else:
                    return DataCollectionResult(
                        ticker=config.ticker,
                        success=False,
                        data_type='market',
                        start_date=config.start_date,
                        end_date=config.end_date,
                        duration_seconds=0,
                        errors=["No market data collected"]
                    )
                    
        except Exception as e:
            logger.error(f"Market data collection failed for {config.ticker}: {e}")
            raise
    
    def _create_date_chunks(self, start_date: date, end_date: date, chunk_size: int) -> List[tuple]:
        """Create date chunks for parallel processing."""
        chunks = []
        current_date = start_date
        
        while current_date <= end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size), end_date)
            chunks.append((current_date, chunk_end))
            current_date = chunk_end + timedelta(days=1)
        
        return chunks
    
    def _estimate_trading_days(self, start_date: date, end_date: date) -> int:
        """Estimate number of trading days between dates."""
        # Rough estimation: 252 trading days per year
        days_diff = (end_date - start_date).days
        years = days_diff / 365.25
        return int(years * 252)
    
    def _estimate_fundamental_periods(self, start_date: date, end_date: date) -> int:
        """Estimate number of fundamental data periods."""
        # 4 quarters per year + 1 annual = 5 periods per year
        days_diff = (end_date - start_date).days
        years = days_diff / 365.25
        return int(years * 5)
    
    async def collect_multiple_tickers(self, tickers: List[str], start_date: str, end_date: str, 
                                     data_types: List[str] = None) -> Dict[str, List[DataCollectionResult]]:
        """
        Collect data for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for collection
            end_date: End date for collection
            data_types: Types of data to collect
            
        Returns:
            Dictionary mapping tickers to their collection results
        """
        if data_types is None:
            data_types = ['technical', 'fundamental']
        
        logger.info(f"Starting data collection for {len(tickers)} tickers")
        
        # Create configurations for all tickers
        configs = []
        for ticker in tickers:
            config = DataCollectionConfig(
                ticker=ticker,
                start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
                end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
                data_types=data_types,
                parallel=True,
                max_workers=self.max_workers
            )
            configs.append(config)
        
        # Collect data for all tickers in parallel
        tasks = [self.collect_historical_data(config) for config in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results by ticker
        organized_results = {}
        for i, result in enumerate(results):
            ticker = tickers[i]
            if isinstance(result, Exception):
                logger.error(f"Collection failed for {ticker}: {result}")
                organized_results[ticker] = []
            else:
                organized_results[ticker] = result
        
        return organized_results
    
    def get_collection_summary(self, results: Dict[str, List[DataCollectionResult]]) -> Dict[str, Any]:
        """Generate summary of collection results."""
        summary = {
            'total_tickers': len(results),
            'successful_tickers': 0,
            'failed_tickers': 0,
            'total_records_collected': 0,
            'total_records_expected': 0,
            'data_type_summary': {},
            'errors': []
        }
        
        for ticker, ticker_results in results.items():
            ticker_success = any(result.success for result in ticker_results)
            if ticker_success:
                summary['successful_tickers'] += 1
            else:
                summary['failed_tickers'] += 1
                summary['errors'].append(f"Collection failed for {ticker}")
            
            for result in ticker_results:
                summary['total_records_collected'] += result.records_collected
                summary['total_records_expected'] += result.records_expected
                
                if result.data_type not in summary['data_type_summary']:
                    summary['data_type_summary'][result.data_type] = {
                        'collected': 0,
                        'expected': 0,
                        'success_rate': 0.0
                    }
                
                summary['data_type_summary'][result.data_type]['collected'] += result.records_collected
                summary['data_type_summary'][result.data_type]['expected'] += result.records_expected
        
        # Calculate success rates
        for data_type in summary['data_type_summary']:
            expected = summary['data_type_summary'][data_type]['expected']
            collected = summary['data_type_summary'][data_type]['collected']
            if expected > 0:
                summary['data_type_summary'][data_type]['success_rate'] = (collected / expected) * 100
        
        return summary 