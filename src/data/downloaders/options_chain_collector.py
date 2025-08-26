#!/usr/bin/env python3
"""
Options Chain Data Collector
Collects 1-minute interval options chain data and futures spot prices for NIFTY and BANKNIFTY
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

import time
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from loguru import logger
import duckdb

from src.nsedata.NseUtility import NseUtils
from src.data.database.options_db_manager import OptionsDatabaseManager


class OptionsChainCollector:
    """Collects options chain data and futures spot prices at 1-minute intervals."""
    
    def __init__(self, db_path: str = "data/options_chain_data.duckdb"):
        """
        Initialize the options chain collector.
        
        Args:
            db_path: Path to DuckDB database for options data
        """
        self.db_path = db_path
        self.nse = NseUtils()
        self.db_manager = OptionsDatabaseManager(db_path)
        self.connection = self.db_manager.connection
        
        # Trading hours (IST)
        self.market_open = "09:30"
        self.market_close = "15:30"
        
        # Indices to track
        self.indices = ['NIFTY', 'BANKNIFTY']
        
        # Trading holidays cache
        self.trading_holidays = None
        self.last_holiday_update = None
        
        logger.info("🚀 Options Chain Collector initialized")
        
    def _get_trading_holidays(self) -> List[str]:
        """Get trading holidays from NSE."""
        try:
            # Update holidays cache once per day
            today = date.today()
            if (self.trading_holidays is None or 
                self.last_holiday_update is None or 
                self.last_holiday_update != today):
                
                self.trading_holidays = self.nse.trading_holidays(list_only=True)
                self.last_holiday_update = today
                logger.info(f"📅 Updated trading holidays: {len(self.trading_holidays)} holidays")
                
            return self.trading_holidays
        except Exception as e:
            logger.error(f"❌ Error fetching trading holidays: {e}")
            return []
    
    def _is_trading_day(self) -> bool:
        """Check if today is a trading day."""
        try:
            today = date.today()
            trading_holidays = self._get_trading_holidays()
            
            # Check if today is weekend
            if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if today is a trading holiday
            today_str = today.strftime('%Y-%m-%d')
            if today_str in trading_holidays:
                return False
                
            return True
        except Exception as e:
            logger.error(f"❌ Error checking trading day: {e}")
            return False
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        try:
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            
            # Check if within market hours
            if self.market_open <= current_time <= self.market_close:
                return True
                
            return False
        except Exception as e:
            logger.error(f"❌ Error checking market hours: {e}")
            return False
    
    def _get_futures_spot_price(self, index: str) -> Optional[float]:
        """Get spot price from futures data."""
        try:
            futures_data = self.nse.futures_data(index, indices=True)
            
            if futures_data is not None and not futures_data.empty:
                # Get the first row (current month contract) for spot price
                if 'lastPrice' in futures_data.columns:
                    spot_price = float(futures_data['lastPrice'].iloc[0])
                    logger.debug(f"✅ Futures spot price for {index}: ₹{spot_price:,.2f}")
                    return spot_price
                elif 'LTP' in futures_data.columns:
                    spot_price = float(futures_data['LTP'].iloc[0])
                    logger.debug(f"✅ Futures LTP for {index}: ₹{spot_price:,.2f}")
                    return spot_price
                else:
                    logger.warning(f"⚠️ No price column found in futures data for {index}")
                    return None
            else:
                logger.warning(f"⚠️ No futures data available for {index}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting futures spot price for {index}: {e}")
            return None
    
    def _process_options_data(self, options_data: pd.DataFrame, index: str, spot_price: float) -> List[Dict]:
        """Process options data and convert to database format."""
        try:
            records = []
            timestamp = datetime.now()
            
            if options_data is None or options_data.empty:
                logger.warning(f"⚠️ No options data for {index}")
                return records
            
            # Calculate ATM strike
            strikes = sorted(options_data['Strike_Price'].unique())
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            # Calculate PCR (Put-Call Ratio)
            atm_data = options_data[options_data['Strike_Price'] == atm_strike]
            if not atm_data.empty:
                call_oi = float(atm_data['CALLS_OI'].iloc[0]) if 'CALLS_OI' in atm_data.columns else 0
                put_oi = float(atm_data['PUTS_OI'].iloc[0]) if 'PUTS_OI' in atm_data.columns else 0
                pcr = put_oi / call_oi if call_oi > 0 else 0
            else:
                pcr = 0
            
            # Process each option contract
            for _, row in options_data.iterrows():
                strike_price = float(row['Strike_Price'])
                expiry_date = pd.to_datetime(row.get('Expiry_Date', '')).date() if pd.notna(row.get('Expiry_Date', '')) else None
                
                # Process CALL options
                if 'CALLS_LTP' in row and pd.notna(row['CALLS_LTP']) and float(row['CALLS_LTP']) > 0:
                    call_record = {
                        'timestamp': timestamp,
                        'index_symbol': index,
                        'strike_price': strike_price,
                        'expiry_date': expiry_date,
                        'option_type': 'CE',
                        'last_price': float(row['CALLS_LTP']) if pd.notna(row['CALLS_LTP']) else None,
                        'bid_price': float(row['CALLS_Bid_Price']) if 'CALLS_Bid_Price' in row and pd.notna(row['CALLS_Bid_Price']) else None,
                        'ask_price': float(row['CALLS_Ask_Price']) if 'CALLS_Ask_Price' in row and pd.notna(row['CALLS_Ask_Price']) else None,
                        'volume': int(row['CALLS_Volume']) if 'CALLS_Volume' in row and pd.notna(row['CALLS_Volume']) else 0,
                        'open_interest': int(row['CALLS_OI']) if 'CALLS_OI' in row and pd.notna(row['CALLS_OI']) else 0,
                        'change_in_oi': int(row['CALLS_Chng_in_OI']) if 'CALLS_Chng_in_OI' in row and pd.notna(row['CALLS_Chng_in_OI']) else 0,
                        'implied_volatility': float(row['CALLS_IV']) if 'CALLS_IV' in row and pd.notna(row['CALLS_IV']) else None,
                        'delta': None,  # Will be calculated later if needed
                        'gamma': None,
                        'theta': None,
                        'vega': None,
                        'spot_price': spot_price,
                        'atm_strike': atm_strike,
                        'pcr': pcr,
                        'created_at': timestamp
                    }
                    records.append(call_record)
                
                # Process PUT options
                if 'PUTS_LTP' in row and pd.notna(row['PUTS_LTP']) and float(row['PUTS_LTP']) > 0:
                    put_record = {
                        'timestamp': timestamp,
                        'index_symbol': index,
                        'strike_price': strike_price,
                        'expiry_date': expiry_date,
                        'option_type': 'PE',
                        'last_price': float(row['PUTS_LTP']) if pd.notna(row['PUTS_LTP']) else None,
                        'bid_price': float(row['PUTS_Bid_Price']) if 'PUTS_Bid_Price' in row and pd.notna(row['PUTS_Bid_Price']) else None,
                        'ask_price': float(row['PUTS_Ask_Price']) if 'PUTS_Ask_Price' in row and pd.notna(row['PUTS_Ask_Price']) else None,
                        'volume': int(row['PUTS_Volume']) if 'PUTS_Volume' in row and pd.notna(row['PUTS_Volume']) else 0,
                        'open_interest': int(row['PUTS_OI']) if 'PUTS_OI' in row and pd.notna(row['PUTS_OI']) else 0,
                        'change_in_oi': int(row['PUTS_Chng_in_OI']) if 'PUTS_Chng_in_OI' in row and pd.notna(row['PUTS_Chng_in_OI']) else 0,
                        'implied_volatility': float(row['PUTS_IV']) if 'PUTS_IV' in row and pd.notna(row['PUTS_IV']) else None,
                        'delta': None,  # Will be calculated later if needed
                        'gamma': None,
                        'theta': None,
                        'vega': None,
                        'spot_price': spot_price,
                        'atm_strike': atm_strike,
                        'pcr': pcr,
                        'created_at': timestamp
                    }
                    records.append(put_record)
            
            logger.info(f"📊 Processed {len(records)} option records for {index}")
            return records
            
        except Exception as e:
            logger.error(f"❌ Error processing options data for {index}: {e}")
            return []
    
    def _insert_options_data(self, records: List[Dict]) -> bool:
        """Insert options data into database."""
        try:
            if not records:
                return True
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Use the options database manager to insert data
            self.db_manager.insert_options_data(df)
            
            logger.info(f"✅ Inserted {len(records)} options records into database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inserting options data: {e}")
            return False
    
    def collect_data_for_index(self, index: str) -> bool:
        """Collect options chain data for a specific index."""
        try:
            logger.info(f"🎯 Collecting data for {index} at {datetime.now().strftime('%H:%M:%S')}")
            
            # Get futures spot price
            spot_price = self._get_futures_spot_price(index)
            if spot_price is None:
                logger.error(f"❌ Could not get spot price for {index}")
                return False
            
            # Get options chain data
            options_data = self.nse.get_live_option_chain(index, indices=True)
            if options_data is None or options_data.empty:
                logger.error(f"❌ No options data for {index}")
                return False
            
            # Process and insert data
            records = self._process_options_data(options_data, index, spot_price)
            success = self._insert_options_data(records)
            
            if success:
                logger.info(f"✅ Successfully collected data for {index} - Spot: ₹{spot_price:,.2f}, Records: {len(records)}")
            else:
                logger.error(f"❌ Failed to insert data for {index}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error collecting data for {index}: {e}")
            return False
    
    def collect_all_data(self) -> bool:
        """Collect data for all indices."""
        try:
            if not self._is_trading_day():
                logger.info("📅 Not a trading day, skipping data collection")
                return True
            
            if not self._is_market_hours():
                logger.info("⏰ Outside market hours, skipping data collection")
                return True
            
            logger.info("🚀 Starting data collection for all indices...")
            
            success_count = 0
            for index in self.indices:
                if self.collect_data_for_index(index):
                    success_count += 1
                else:
                    logger.warning(f"⚠️ Failed to collect data for {index}")
                
                # Small delay between indices to avoid rate limiting
                time.sleep(2)
            
            logger.info(f"📊 Data collection completed: {success_count}/{len(self.indices)} indices successful")
            return success_count == len(self.indices)
            
        except Exception as e:
            logger.error(f"❌ Error in collect_all_data: {e}")
            return False
    
    def start_scheduler(self):
        """Start the scheduler to collect data every 3 minutes during market hours."""
        logger.info("🚀 Starting Options Chain Data Collector Scheduler")
        logger.info(f"📅 Trading hours: {self.market_open} - {self.market_close} IST")
        logger.info(f"📊 Indices: {', '.join(self.indices)}")
        logger.info("⏰ Collection interval: 3 minutes")
        
        # Schedule data collection every 3 minutes
        schedule.every(3).minutes.do(self.collect_all_data)
        
        # Run initial collection
        logger.info("📊 Running initial data collection...")
        self.collect_all_data()
        
        # Start scheduler loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                logger.info("🛑 Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_recent_data(self, index: str, minutes: int = 60) -> pd.DataFrame:
        """Get recent options data for analysis."""
        try:
            query = """
                SELECT * FROM options_chain_data 
                WHERE index_symbol = ? 
                AND timestamp >= NOW() - INTERVAL '{} minutes'
                ORDER BY timestamp DESC, strike_price, option_type
            """.format(minutes)
            
            df = self.connection.execute(query, [index]).fetchdf()
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting recent data for {index}: {e}")
            return pd.DataFrame()
    
    def get_daily_summary(self, index: str, date: str = None) -> Dict:
        """Get daily summary statistics for an index."""
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT strike_price) as unique_strikes,
                    AVG(spot_price) as avg_spot_price,
                    MAX(spot_price) as max_spot_price,
                    MIN(spot_price) as min_spot_price,
                    AVG(pcr) as avg_pcr,
                    MAX(pcr) as max_pcr,
                    MIN(pcr) as min_pcr
                FROM options_chain_data 
                WHERE index_symbol = ? 
                AND DATE(timestamp) = ?
            """
            
            result = self.connection.execute(query, [index, date]).fetchone()
            
            if result:
                return {
                    'index': index,
                    'date': date,
                    'total_records': result[0],
                    'unique_strikes': result[1],
                    'avg_spot_price': result[2],
                    'max_spot_price': result[3],
                    'min_spot_price': result[4],
                    'avg_pcr': result[5],
                    'max_pcr': result[6],
                    'min_pcr': result[7]
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting daily summary for {index}: {e}")
            return {}


def main():
    """Main function to run the options chain collector."""
    try:
        # Initialize collector
        collector = OptionsChainCollector()
        
        # Start scheduler
        collector.start_scheduler()
        
    except KeyboardInterrupt:
        logger.info("🛑 Options Chain Collector stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in Options Chain Collector: {e}")


if __name__ == "__main__":
    main()
