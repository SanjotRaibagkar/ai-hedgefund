#!/usr/bin/env python3
"""
Enhanced Options Chain Collector V2
Two-process approach:
1. Process 1: Collect 1-minute data and write to parquet files
2. Process 2: Batch process parquet files to DuckDB every 5 minutes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging
from typing import Dict, List, Optional, Any
import glob
import shutil

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.nsedata.NseUtility import NseUtils

class OptionsChainCollectorV2:
    """Enhanced Options Chain Collector with parquet-based approach."""
    
    def __init__(self):
        """Initialize the collector."""
        self.logger = logging.getLogger(__name__)
        self.nse = NseUtils()
        
        # Create directories
        self.parquet_dir = "data/options_parquet"
        self.processed_dir = "data/options_processed"
        self.temp_dir = "data/options_temp"
        
        for directory in [self.parquet_dir, self.processed_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Configuration
        self.collection_interval = 60  # 1 minute
        self.batch_interval = 300  # 5 minutes
        self.indices = ['NIFTY', 'BANKNIFTY']
        
        self.logger.info("🚀 Options Chain Collector V2 initialized")
        self.logger.info(f"   Collection interval: {self.collection_interval}s")
        self.logger.info(f"   Batch interval: {self.batch_interval}s")
        self.logger.info(f"   Indices: {self.indices}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in format for filenames."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _get_date_folder(self) -> str:
        """Get date folder name."""
        return datetime.now().strftime('%Y%m%d')
    
    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours."""
        now = datetime.now()
        current_time = now.time()
        
        # Market hours: 9:15 AM to 3:30 PM (Monday to Friday)
        market_start = datetime.strptime('09:15:00', '%H:%M:%S').time()
        market_end = datetime.strptime('15:30:00', '%H:%M:%S').time()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's during market hours
        return market_start <= current_time <= market_end
    
    def collect_options_data(self) -> Dict[str, Any]:
        """Collect options data for all indices and save to parquet."""
        if not self._is_market_hours():
            self.logger.info("⏰ Outside market hours, skipping collection")
            return {}
        
        timestamp = self._get_timestamp()
        date_folder = self._get_date_folder()
        
        # Create date-specific directory
        daily_parquet_dir = os.path.join(self.parquet_dir, date_folder)
        os.makedirs(daily_parquet_dir, exist_ok=True)
        
        collected_data = {}
        
        for index in self.indices:
            try:
                self.logger.info(f"📊 Collecting {index} options data...")
                
                # Get options chain data
                options_data = self.nse.get_live_option_chain(index, indices=True)
                
                if options_data is not None and isinstance(options_data, pd.DataFrame) and not options_data.empty:
                    # Add metadata
                    options_data['collection_timestamp'] = timestamp
                    options_data['index_name'] = index
                    options_data['date'] = datetime.now().date()
                    
                    # Save to parquet file
                    filename = f"{index}_{timestamp}.parquet"
                    filepath = os.path.join(daily_parquet_dir, filename)
                    
                    options_data.to_parquet(filepath, index=False, compression='snappy')
                    
                    collected_data[index] = {
                        'filepath': filepath,
                        'rows': len(options_data),
                        'timestamp': timestamp,
                        'success': True
                    }
                    
                    self.logger.info(f"✅ {index}: {len(options_data)} rows saved to {filename}")
                    
                else:
                    self.logger.warning(f"⚠️ No data received for {index}")
                    collected_data[index] = {
                        'success': False,
                        'error': 'No data received'
                    }
                    
            except Exception as e:
                self.logger.error(f"❌ Error collecting {index} data: {e}")
                collected_data[index] = {
                    'success': False,
                    'error': str(e)
                }
        
        return collected_data
    
    def process_parquet_to_duckdb(self) -> Dict[str, Any]:
        """Process parquet files to DuckDB in batches."""
        self.logger.info("🔄 Starting batch processing of parquet files to DuckDB")
        
        # Get today's date folder
        date_folder = self._get_date_folder()
        daily_parquet_dir = os.path.join(self.parquet_dir, date_folder)
        
        if not os.path.exists(daily_parquet_dir):
            self.logger.warning(f"⚠️ No parquet directory found for {date_folder}")
            return {'processed': 0, 'errors': 0}
        
        # Get all parquet files
        parquet_files = glob.glob(os.path.join(daily_parquet_dir, "*.parquet"))
        
        if not parquet_files:
            self.logger.info("📁 No parquet files to process")
            return {'processed': 0, 'errors': 0}
        
        self.logger.info(f"📁 Found {len(parquet_files)} parquet files to process")
        
        processed_count = 0
        error_count = 0
        total_rows = 0
        
        for filepath in parquet_files:
            try:
                # Read parquet file
                df = pd.read_parquet(filepath)
                
                if df.empty:
                    self.logger.warning(f"⚠️ Empty file: {os.path.basename(filepath)}")
                    continue
                
                # Process data for DuckDB
                processed_df = self._prepare_data_for_duckdb(df)
                
                if processed_df is not None and not processed_df.empty:
                    # Insert into DuckDB
                    success = self._insert_to_duckdb(processed_df)
                    
                    if success:
                        processed_count += 1
                        total_rows += len(processed_df)
                        
                        # Move to processed directory
                        processed_filepath = os.path.join(self.processed_dir, os.path.basename(filepath))
                        shutil.move(filepath, processed_filepath)
                        
                        self.logger.info(f"✅ Processed: {os.path.basename(filepath)} ({len(processed_df)} rows)")
                    else:
                        error_count += 1
                        self.logger.error(f"❌ Failed to insert: {os.path.basename(filepath)}")
                else:
                    error_count += 1
                    self.logger.warning(f"⚠️ No valid data in: {os.path.basename(filepath)}")
                    
            except Exception as e:
                error_count += 1
                self.logger.error(f"❌ Error processing {os.path.basename(filepath)}: {e}")
        
        self.logger.info(f"📊 Batch processing completed:")
        self.logger.info(f"   Processed files: {processed_count}")
        self.logger.info(f"   Total rows: {total_rows}")
        self.logger.info(f"   Errors: {error_count}")
        
        return {
            'processed': processed_count,
            'total_rows': total_rows,
            'errors': error_count
        }
    
    def _prepare_data_for_duckdb(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare data for DuckDB insertion - transform from wide to long format."""
        try:
            # Ensure required columns exist (using actual parquet file column names)
            required_columns = ['Strike_Price', 'CALLS_OI', 'PUTS_OI', 'CALLS_Volume', 'PUTS_Volume', 'index_name']
            
            for col in required_columns:
                if col not in df.columns:
                    self.logger.warning(f"⚠️ Missing column: {col}")
                    return None
            
            # Clean and prepare data
            processed_df = df.copy()
            
            # Convert numeric columns (using actual column names)
            numeric_columns = ['Strike_Price', 'CALLS_OI', 'PUTS_OI', 'CALLS_Volume', 'PUTS_Volume', 
                             'CALLS_Chng_in_OI', 'PUTS_Chng_in_OI', 'CALLS_IV', 'PUTS_IV',
                             'CALLS_LTP', 'PUTS_LTP', 'CALLS_Bid_Price', 'PUTS_Bid_Price',
                             'CALLS_Ask_Price', 'PUTS_Ask_Price']
            
            for col in numeric_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # Add timestamp columns
            if 'collection_timestamp' in processed_df.columns:
                processed_df['timestamp'] = pd.to_datetime(processed_df['collection_timestamp'], 
                                                         format='%Y%m%d_%H%M%S')
            else:
                processed_df['timestamp'] = datetime.now()
            
            # Transform from wide to long format (separate CALL and PUT rows)
            call_data = []
            put_data = []
            
            for _, row in processed_df.iterrows():
                # CALL option data
                call_data.append({
                    'timestamp': row['timestamp'],
                    'index_symbol': row['index_name'],
                    'strike_price': row['Strike_Price'],
                    'expiry_date': pd.to_datetime(row['Expiry_Date'], format='%d-%b-%Y').date(),
                    'option_type': 'CE',
                    'last_price': row.get('CALLS_LTP', 0),
                    'bid_price': row.get('CALLS_Bid_Price', 0),
                    'ask_price': row.get('CALLS_Ask_Price', 0),
                    'volume': row.get('CALLS_Volume', 0),
                    'open_interest': row.get('CALLS_OI', 0),
                    'change_in_oi': row.get('CALLS_Chng_in_OI', 0),
                    'implied_volatility': row.get('CALLS_IV', 0),
                    'delta': 0,  # Not available in current data
                    'gamma': 0,  # Not available in current data
                    'theta': 0,  # Not available in current data
                    'vega': 0,   # Not available in current data
                    'spot_price': 0,  # Will be calculated later
                    'atm_strike': 0,  # Will be calculated later
                    'pcr': 0,    # Will be calculated later
                    'created_at': row['timestamp']
                })
                
                # PUT option data
                put_data.append({
                    'timestamp': row['timestamp'],
                    'index_symbol': row['index_name'],
                    'strike_price': row['Strike_Price'],
                    'expiry_date': pd.to_datetime(row['Expiry_Date'], format='%d-%b-%Y').date(),
                    'option_type': 'PE',
                    'last_price': row.get('PUTS_LTP', 0),
                    'bid_price': row.get('PUTS_Bid_Price', 0),
                    'ask_price': row.get('PUTS_Ask_Price', 0),
                    'volume': row.get('PUTS_Volume', 0),
                    'open_interest': row.get('PUTS_OI', 0),
                    'change_in_oi': row.get('PUTS_Chng_in_OI', 0),
                    'implied_volatility': row.get('PUTS_IV', 0),
                    'delta': 0,  # Not available in current data
                    'gamma': 0,  # Not available in current data
                    'theta': 0,  # Not available in current data
                    'vega': 0,   # Not available in current data
                    'spot_price': 0,  # Will be calculated later
                    'atm_strike': 0,  # Will be calculated later
                    'pcr': 0,    # Will be calculated later
                    'created_at': row['timestamp']
                })
            
            # Combine CALL and PUT data
            combined_data = call_data + put_data
            result_df = pd.DataFrame(combined_data)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing data: {e}")
            return None
    
    def _insert_to_duckdb(self, df: pd.DataFrame) -> bool:
        """Insert data into DuckDB."""
        try:
            # Import here to avoid multiprocessing issues
            from src.data.database.options_db_manager import OptionsDatabaseManager
            
            # Create a new options database manager instance
            db_manager = OptionsDatabaseManager()
            
            # Use the options database manager to insert data
            success = db_manager.insert_options_data(df)
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error inserting to DuckDB: {e}")
            return False
    
    def run_collection_loop(self):
        """Run the collection loop (Process 1)."""
        self.logger.info("🚀 Starting Options Chain Collection Loop (Process 1)")
        self.logger.info(f"   Collection interval: {self.collection_interval} seconds")
        
        while True:
            try:
                if self._is_market_hours():
                    # Collect data
                    results = self.collect_options_data()
                    
                    # Log results
                    for index, result in results.items():
                        if result.get('success'):
                            self.logger.info(f"✅ {index}: {result.get('rows', 0)} rows collected")
                        else:
                            self.logger.warning(f"⚠️ {index}: {result.get('error', 'Unknown error')}")
                else:
                    self.logger.info("⏰ Outside market hours, waiting...")
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Collection loop stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def run_batch_processing_loop(self):
        """Run the batch processing loop (Process 2)."""
        self.logger.info("🔄 Starting Batch Processing Loop (Process 2)")
        self.logger.info(f"   Batch interval: {self.batch_interval} seconds")
        
        while True:
            try:
                if self._is_market_hours():
                    # Process parquet files to DuckDB
                    results = self.process_parquet_to_duckdb()
                    
                    self.logger.info(f"📊 Batch processing: {results['processed']} files, "
                                   f"{results['total_rows']} rows, {results['errors']} errors")
                else:
                    self.logger.info("⏰ Outside market hours, waiting...")
                
                # Wait for next batch processing
                time.sleep(self.batch_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Batch processing loop stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in batch processing loop: {e}")
                time.sleep(self.batch_interval)


def main():
    """Main function for testing."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    collector = OptionsChainCollectorV2()
    
    # Test collection
    print("🧪 Testing collection...")
    results = collector.collect_options_data()
    print(f"Collection results: {results}")
    
    # Test batch processing
    print("\n🧪 Testing batch processing...")
    batch_results = collector.process_parquet_to_duckdb()
    print(f"Batch results: {batch_results}")


if __name__ == "__main__":
    main()
