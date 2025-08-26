"""
Simple Price Updater for AI Hedge Fund
Uses NSE utility to update price data daily without external APIs.
"""

import duckdb
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import sys
import os

# Add src to path
sys.path.append('./src')

from nsedata.NseUtility import NseUtils


class SimplePriceUpdater:
    """Simple price updater using NSE utility."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        """
        Initialize simple price updater.
        
        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = db_path
        self.nse = NseUtils()
        self.conn = duckdb.connect(db_path)
        
        logger.info("Simple Price Updater initialized")
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols from price_data table."""
        try:
            symbols = self.conn.execute("SELECT DISTINCT symbol FROM price_data").fetchdf()
            return symbols['symbol'].tolist()
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def get_latest_date_for_symbol(self, symbol: str) -> Optional[str]:
        """Get the latest date for a symbol in price_data table."""
        try:
            result = self.conn.execute(
                "SELECT MAX(date) as latest_date FROM price_data WHERE symbol = ?", 
                [symbol]
            ).fetchone()
            
            if result and result[0]:
                return result[0].strftime('%Y-%m-%d')
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest date for {symbol}: {e}")
            return None
    
    def get_missing_dates(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """Get missing dates between start_date and end_date for a symbol."""
        try:
            # Get existing dates for the symbol
            existing_dates = self.conn.execute(
                "SELECT date FROM price_data WHERE symbol = ? AND date BETWEEN ? AND ?",
                [symbol, start_date, end_date]
            ).fetchdf()
            
            if existing_dates.empty:
                # No existing data, return all dates
                return self._generate_date_range(start_date, end_date)
            
            existing_dates_list = [d.strftime('%Y-%m-%d') for d in existing_dates['date']]
            
            # Generate all dates in range
            all_dates = self._generate_date_range(start_date, end_date)
            
            # Return missing dates
            missing_dates = [d for d in all_dates if d not in existing_dates_list]
            
            return missing_dates
            
        except Exception as e:
            logger.error(f"Failed to get missing dates for {symbol}: {e}")
            return []
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Generate list of dates between start_date and end_date."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return dates
    
    def get_bhavcopy_data(self, trade_date: str) -> Optional[pd.DataFrame]:
        """Get bhavcopy data for a specific date."""
        try:
            # Convert YYYY-MM-DD to DD-MM-YYYY for NSE API
            date_obj = datetime.strptime(trade_date, '%Y-%m-%d')
            nse_date = date_obj.strftime('%d-%m-%Y')
            
            logger.info(f"Fetching bhavcopy for {nse_date}")
            df = self.nse.equity_bhav_copy(nse_date)
            
            if df is not None and not df.empty:
                # Standardize column names
                df.columns = [col.strip() for col in df.columns]
                
                # Select and rename relevant columns
                column_mapping = {
                    'TckrSymb': 'symbol',
                    'SctySrs': 'series',
                    'OpnPric': 'open_price',
                    'HghPric': 'high_price',
                    'LwPric': 'low_price',
                    'ClsPric': 'close_price',
                    'LastPric': 'last_price',
                    'PrvsClsgPric': 'prev_close',
                    'TtlTradgVol': 'volume',
                    'TtlTrfVal': 'turnover',
                    'TradDt': 'date'
                }
                
                # Filter columns that exist
                available_columns = [col for col in column_mapping.keys() if col in df.columns]
                df = df[available_columns].copy()
                
                # Rename columns
                df = df.rename(columns={col: column_mapping[col] for col in available_columns})
                
                # Convert date column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                
                # Convert numeric columns
                numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'last_price', 'prev_close', 'volume', 'turnover']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
                
                # Add metadata
                df['last_updated'] = datetime.now()
                
                logger.info(f"Successfully fetched bhavcopy: {len(df)} records")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get bhavcopy for {trade_date}: {e}")
            return None
    
    def update_symbol_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Update data for a specific symbol."""
        try:
            logger.info(f"Updating data for {symbol} from {start_date} to {end_date}")
            
            # Get missing dates
            missing_dates = self.get_missing_dates(symbol, start_date, end_date)
            
            if not missing_dates:
                logger.info(f"No missing dates for {symbol}")
                return {"success": True, "symbol": symbol, "records_added": 0, "message": "No missing dates"}
            
            logger.info(f"Missing dates for {symbol}: {len(missing_dates)} dates")
            
            total_records = 0
            
            for date in missing_dates:
                try:
                    # Get bhavcopy data for this date
                    bhavcopy_df = self.get_bhavcopy_data(date)
                    
                    if bhavcopy_df is not None and not bhavcopy_df.empty:
                        # Filter for this symbol
                        symbol_data = bhavcopy_df[bhavcopy_df['symbol'] == symbol]
                        
                        if not symbol_data.empty:
                            # Insert into database
                            self.conn.execute("""
                                INSERT INTO price_data (symbol, date, open_price, high_price, low_price, close_price, volume, turnover, last_updated)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, [
                                str(symbol_data.iloc[0]['symbol']),
                                symbol_data.iloc[0]['date'],
                                float(symbol_data.iloc[0].get('open_price', 0)),
                                float(symbol_data.iloc[0].get('high_price', 0)),
                                float(symbol_data.iloc[0].get('low_price', 0)),
                                float(symbol_data.iloc[0].get('close_price', 0)),
                                float(symbol_data.iloc[0].get('volume', 0)),
                                float(symbol_data.iloc[0].get('turnover', 0)),
                                symbol_data.iloc[0]['last_updated']
                            ])
                            
                            total_records += 1
                            logger.info(f"Added data for {symbol} on {date}")
                        else:
                            logger.warning(f"No data found for {symbol} on {date}")
                    else:
                        logger.warning(f"No bhavcopy data available for {date}")
                        
                except Exception as e:
                    logger.error(f"Failed to update {symbol} for {date}: {e}")
                    continue
            
            return {
                "success": True,
                "symbol": symbol,
                "records_added": total_records,
                "missing_dates": len(missing_dates)
            }
            
        except Exception as e:
            logger.error(f"Failed to update {symbol}: {e}")
            return {"success": False, "symbol": symbol, "error": str(e)}
    
    def _get_missing_dates_for_all_symbols(self, symbols: List[str], target_date: str) -> List[str]:
        """Get all missing dates for all symbols up to target_date."""
        try:
            all_missing_dates = set()
            
            for symbol in symbols:
                # Get latest date for this symbol
                latest_date = self.get_latest_date_for_symbol(symbol)
                
                if latest_date:
                    # Start from day after latest data
                    start_date = (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    # No existing data, start from a reasonable date (e.g., 1 year ago)
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
                # Get missing dates for this symbol
                symbol_missing_dates = self.get_missing_dates(symbol, start_date, target_date)
                all_missing_dates.update(symbol_missing_dates)
            
            # Return sorted list of unique missing dates
            return sorted(list(all_missing_dates))
            
        except Exception as e:
            logger.error(f"Failed to get missing dates for all symbols: {e}")
            return []
    
    def _update_symbols_for_date(self, symbols: List[str], bhavcopy_df: pd.DataFrame, date: str) -> Dict[str, Any]:
        """Update all symbols for a specific date using the bhavcopy data."""
        try:
            records_added = 0
            successful_symbols = set()
            failed_symbols = set()
            
            for symbol in symbols:
                try:
                    # Check if this symbol exists in bhavcopy data
                    symbol_data = bhavcopy_df[bhavcopy_df['symbol'] == symbol]
                    
                    if not symbol_data.empty:
                        # Check if data already exists for this symbol and date
                        existing_data = self.conn.execute(
                            "SELECT COUNT(*) FROM price_data WHERE symbol = ? AND date = ?",
                            [symbol, date]
                        ).fetchone()[0]
                        
                        if existing_data == 0:
                            # Insert new data
                            self.conn.execute("""
                                INSERT INTO price_data (symbol, date, open_price, high_price, low_price, close_price, volume, turnover, last_updated)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, [
                                str(symbol_data.iloc[0]['symbol']),
                                symbol_data.iloc[0]['date'],
                                float(symbol_data.iloc[0].get('open_price', 0)),
                                float(symbol_data.iloc[0].get('high_price', 0)),
                                float(symbol_data.iloc[0].get('low_price', 0)),
                                float(symbol_data.iloc[0].get('close_price', 0)),
                                float(symbol_data.iloc[0].get('volume', 0)),
                                float(symbol_data.iloc[0].get('turnover', 0)),
                                symbol_data.iloc[0]['last_updated']
                            ])
                            
                            records_added += 1
                            successful_symbols.add(symbol)
                            logger.debug(f"Added data for {symbol} on {date}")
                        else:
                            logger.debug(f"Data already exists for {symbol} on {date}")
                    else:
                        logger.debug(f"No data found for {symbol} on {date}")
                        
                except Exception as e:
                    logger.error(f"Failed to update {symbol} for {date}: {e}")
                    failed_symbols.add(symbol)
            
            return {
                "records_added": records_added,
                "successful_symbols": successful_symbols,
                "failed_symbols": failed_symbols
            }
            
        except Exception as e:
            logger.error(f"Failed to update symbols for date {date}: {e}")
            return {
                "records_added": 0,
                "successful_symbols": set(),
                "failed_symbols": set(symbols)
            }
    
    def run_daily_update(self, target_date: str = None) -> Dict[str, Any]:
        """
        Run daily price update for all symbols.
        
        Args:
            target_date: Target date (YYYY-MM-DD). If None, uses yesterday.
            
        Returns:
            Dictionary with update results
        """
        try:
            # Determine target date
            if target_date is None:
                target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"Starting daily price update for {target_date}")
            
            # Get all symbols
            symbols = self.get_all_symbols()
            
            if not symbols:
                logger.warning("No symbols found in database")
                return {"success": False, "message": "No symbols found"}
            
            logger.info(f"Found {len(symbols)} symbols to update")
            
            # Get missing dates for all symbols
            missing_dates = self._get_missing_dates_for_all_symbols(symbols, target_date)
            
            if not missing_dates:
                logger.info("No missing dates found for any symbol")
                return {
                    "success": True,
                    "target_date": target_date,
                    "total_symbols": len(symbols),
                    "successful_symbols": len(symbols),
                    "failed_symbols": 0,
                    "total_records_added": 0,
                    "message": "No missing dates"
                }
            
            logger.info(f"Missing dates to process: {len(missing_dates)}")
            
            # Process each missing date
            total_records = 0
            successful_symbols = set()
            failed_symbols = set()
            
            for date in missing_dates:
                try:
                    logger.info(f"Processing date: {date}")
                    
                    # Download bhavcopy for this date once
                    bhavcopy_df = self.get_bhavcopy_data(date)
                    
                    if bhavcopy_df is not None and not bhavcopy_df.empty:
                        # Update all symbols for this date
                        date_results = self._update_symbols_for_date(symbols, bhavcopy_df, date)
                        
                        total_records += date_results["records_added"]
                        successful_symbols.update(date_results["successful_symbols"])
                        failed_symbols.update(date_results["failed_symbols"])
                        
                        logger.info(f"Date {date}: {date_results['records_added']} records added")
                    else:
                        logger.warning(f"No bhavcopy data available for {date}")
                        
                except Exception as e:
                    logger.error(f"Failed to process date {date}: {e}")
                    continue
            
            # Summary
            summary = {
                "success": True,
                "target_date": target_date,
                "total_symbols": len(symbols),
                "successful_symbols": len(successful_symbols),
                "failed_symbols": len(failed_symbols),
                "total_records_added": total_records,
                "missing_dates_processed": len(missing_dates)
            }
            
            logger.info(f"Daily update completed: {len(successful_symbols)}/{len(symbols)} symbols successful")
            logger.info(f"Total records added: {total_records}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
            return {"success": False, "error": str(e)}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    """Main function to run the simple price updater."""
    print("Simple Price Updater for AI Hedge Fund")
    print("=" * 50)
    
    try:
        updater = SimplePriceUpdater()
        
        # Run daily update
        result = updater.run_daily_update()
        
        if result["success"]:
            print(f"✅ Update completed successfully!")
            print(f"   Target date: {result['target_date']}")
            print(f"   Total symbols: {result['total_symbols']}")
            print(f"   Successful: {result['successful_symbols']}")
            print(f"   Failed: {result['failed_symbols']}")
            print(f"   Records added: {result['total_records_added']}")
        else:
            print(f"❌ Update failed: {result.get('error', 'Unknown error')}")
        
        updater.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
