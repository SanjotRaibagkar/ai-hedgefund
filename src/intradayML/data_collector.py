"""
Intraday Data Collector
Collects all required data for intraday ML prediction using NSE utility.
"""

import sys
import os
sys.path.append('./src')

import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import time

from nsedata.NseUtility import NseUtils


class IntradayDataCollector:
    """Collects all required data for intraday ML prediction."""
    
    def __init__(self, db_path: str = "data/intraday_ml_data.duckdb"):
        """
        Initialize the intraday data collector.
        
        Args:
            db_path: Path to DuckDB database for intraday ML data
        """
        self.db_path = db_path
        self.nse = NseUtils()
        self.conn = duckdb.connect(db_path)
        
        # Initialize database tables
        self._initialize_database()
        
        logger.info("🚀 Intraday Data Collector initialized")
    
    def _initialize_database(self):
        """Initialize database tables for intraday ML data."""
        try:
            # Options Chain Data Table (15-min intervals)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS intraday_options_data (
                    timestamp TIMESTAMP,
                    index_symbol VARCHAR,
                    strike_price DOUBLE,
                    expiry_date DATE,
                    option_type VARCHAR,
                    last_price DOUBLE,
                    bid_price DOUBLE,
                    ask_price DOUBLE,
                    volume BIGINT,
                    open_interest BIGINT,
                    change_in_oi BIGINT,
                    implied_volatility DOUBLE,
                    delta DOUBLE,
                    gamma DOUBLE,
                    theta DOUBLE,
                    vega DOUBLE,
                    spot_price DOUBLE,
                    atm_strike DOUBLE,
                    pcr_oi DOUBLE,
                    pcr_volume DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, index_symbol, strike_price, expiry_date, option_type)
                )
            """)
            
            # Index OHLCV Data Table (15-min intervals)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS intraday_index_data (
                    timestamp TIMESTAMP,
                    index_symbol VARCHAR,
                    open_price DOUBLE,
                    high_price DOUBLE,
                    low_price DOUBLE,
                    close_price DOUBLE,
                    volume BIGINT,
                    turnover DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, index_symbol)
                )
            """)
            
            # FII/DII Data Table (daily)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS intraday_fii_dii_data (
                    date DATE,
                    fii_buy DOUBLE,
                    fii_sell DOUBLE,
                    fii_net DOUBLE,
                    dii_buy DOUBLE,
                    dii_sell DOUBLE,
                    dii_net DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (date)
                )
            """)
            
            # India VIX Data Table (15-min intervals)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS intraday_vix_data (
                    timestamp TIMESTAMP,
                    vix_value DOUBLE,
                    vix_change DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp)
                )
            """)
            
            # Labels Table (15-min intervals)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS intraday_labels (
                    timestamp TIMESTAMP,
                    index_symbol VARCHAR,
                    label INTEGER,  -- 1 for UP, -1 for DOWN
                    future_close DOUBLE,
                    current_close DOUBLE,
                    return_pct DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, index_symbol)
                )
            """)
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info("✅ Intraday ML database tables initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better query performance."""
        try:
            # Options data indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_options_timestamp ON intraday_options_data(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_options_index ON intraday_options_data(index_symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_options_strike ON intraday_options_data(strike_price)")
            
            # Index data indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_index_timestamp ON intraday_index_data(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_index_symbol ON intraday_index_data(index_symbol)")
            
            # FII/DII data indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_fii_dii_date ON intraday_fii_dii_data(date)")
            
            # VIX data indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_vix_timestamp ON intraday_vix_data(timestamp)")
            
            # Labels indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_labels_timestamp ON intraday_labels(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_labels_index ON intraday_labels(index_symbol)")
            
            logger.info("✅ Intraday ML database indexes created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
    
    def collect_options_chain_data(self, index_symbol: str = 'NIFTY') -> pd.DataFrame:
        """
        Collect options chain data for the specified index.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            
        Returns:
            DataFrame with options chain data
        """
        try:
            logger.info(f"📊 Collecting options chain data for {index_symbol}")
            
            # Use get_live_option_chain method for indices (NIFTY/BANKNIFTY)
            options_data = self.nse.get_live_option_chain(index_symbol, indices=True, oi_mode="full")
            
            if options_data is None or options_data.empty:
                logger.warning(f"❌ No options data available for {index_symbol}")
                return pd.DataFrame()
            
            # Process and standardize the data
            processed_data = self._process_live_options_data(options_data, index_symbol)
            
            # Insert into database
            if not processed_data.empty:
                self._insert_options_data(processed_data)
                logger.info(f"✅ Collected {len(processed_data)} options records for {index_symbol}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting options data for {index_symbol}: {e}")
            return pd.DataFrame()
    
    def _process_live_options_data(self, options_data: pd.DataFrame, index_symbol: str) -> pd.DataFrame:
        """Process and standardize live options chain data from get_live_option_chain."""
        try:
            # Get current timestamp
            current_time = datetime.now()
            
            # Get spot price from index details
            try:
                index_details = self.nse.get_index_details(index_symbol)
                if not index_details.empty:
                    spot_price = index_details.iloc[0].get('lastPrice', 0)
                else:
                    spot_price = 0
            except:
                spot_price = 0
            
            # Calculate ATM strike (nearest to spot price)
            atm_strike = self._calculate_atm_strike(spot_price)
            
            # Process each option contract
            processed_records = []
            
            for _, row in options_data.iterrows():
                try:
                    # Extract strike price
                    strike = row.get('Strike_Price', 0)
                    expiry_str = row.get('Expiry_Date', date.today())
                    
                    # Convert date format from "DD-MMM-YYYY" to "YYYY-MM-DD"
                    try:
                        if isinstance(expiry_str, str):
                            expiry = datetime.strptime(expiry_str, '%d-%b-%Y').date()
                        else:
                            expiry = expiry_str
                    except:
                        expiry = date.today()
                    
                    # Process CALL options
                    if row.get('CALLS_OI', 0) > 0:
                        processed_records.append({
                            'timestamp': current_time,
                            'index_symbol': index_symbol,
                            'strike_price': strike,
                            'expiry_date': expiry,
                            'option_type': 'CE',
                            'last_price': row.get('CALLS_LTP', 0),
                            'bid_price': row.get('CALLS_Bid_Price', 0),
                            'ask_price': row.get('CALLS_Ask_Price', 0),
                            'volume': row.get('CALLS_Volume', 0),
                            'open_interest': row.get('CALLS_OI', 0),
                            'change_in_oi': row.get('CALLS_Chng_in_OI', 0),
                            'implied_volatility': row.get('CALLS_IV', 0),
                            'delta': 0,  # Not available in live data
                            'gamma': 0,  # Not available in live data
                            'theta': 0,  # Not available in live data
                            'vega': 0,   # Not available in live data
                            'spot_price': spot_price,
                            'atm_strike': atm_strike,
                            'pcr_oi': 0,  # Will calculate separately
                            'pcr_volume': 0  # Will calculate separately
                        })
                    
                    # Process PUT options
                    if row.get('PUTS_OI', 0) > 0:
                        processed_records.append({
                            'timestamp': current_time,
                            'index_symbol': index_symbol,
                            'strike_price': strike,
                            'expiry_date': expiry,
                            'option_type': 'PE',
                            'last_price': row.get('PUTS_LTP', 0),
                            'bid_price': row.get('PUTS_Bid_Price', 0),
                            'ask_price': row.get('PUTS_Ask_Price', 0),
                            'volume': row.get('PUTS_Volume', 0),
                            'open_interest': row.get('PUTS_OI', 0),
                            'change_in_oi': row.get('PUTS_Chng_in_OI', 0),
                            'implied_volatility': row.get('PUTS_IV', 0),
                            'delta': 0,  # Not available in live data
                            'gamma': 0,  # Not available in live data
                            'theta': 0,  # Not available in live data
                            'vega': 0,   # Not available in live data
                            'spot_price': spot_price,
                            'atm_strike': atm_strike,
                            'pcr_oi': 0,  # Will calculate separately
                            'pcr_volume': 0  # Will calculate separately
                        })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing option record: {e}")
                    continue
            
            # Calculate PCR for each strike
            if processed_records:
                df = pd.DataFrame(processed_records)
                df = self._calculate_pcr_for_live_data(df)
                return df
            
            return pd.DataFrame(processed_records)
            
        except Exception as e:
            logger.error(f"❌ Error processing live options data: {e}")
            return pd.DataFrame()
    
    def _calculate_pcr_for_live_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PCR for live options data."""
        try:
            # Group by strike price and calculate PCR
            for strike in df['strike_price'].unique():
                strike_data = df[df['strike_price'] == strike]
                
                ce_oi = strike_data[strike_data['option_type'] == 'CE']['open_interest'].sum()
                pe_oi = strike_data[strike_data['option_type'] == 'PE']['open_interest'].sum()
                ce_volume = strike_data[strike_data['option_type'] == 'CE']['volume'].sum()
                pe_volume = strike_data[strike_data['option_type'] == 'PE']['volume'].sum()
                
                # Calculate PCR
                pcr_oi = pe_oi / ce_oi if ce_oi > 0 else 0
                pcr_volume = pe_volume / ce_volume if ce_volume > 0 else 0
                
                # Update PCR values
                df.loc[df['strike_price'] == strike, 'pcr_oi'] = pcr_oi
                df.loc[df['strike_price'] == strike, 'pcr_volume'] = pcr_volume
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating PCR for live data: {e}")
            return df
    
    def _calculate_atm_strike(self, spot_price: float) -> float:
        """Calculate ATM strike price (nearest to spot price)."""
        try:
            # For NIFTY, strikes are in multiples of 50
            # For BANKNIFTY, strikes are in multiples of 100
            if spot_price <= 10000:  # NIFTY
                strike_multiple = 50
            else:  # BANKNIFTY
                strike_multiple = 100
            
            atm_strike = round(spot_price / strike_multiple) * strike_multiple
            return atm_strike
            
        except Exception as e:
            logger.error(f"❌ Error calculating ATM strike: {e}")
            return spot_price
    
    def _calculate_pcr_for_strike(self, options_data: pd.DataFrame, strike: float, metric: str) -> float:
        """Calculate PCR for a specific strike price."""
        try:
            # Filter data for this strike - handle different column names
            strike_col = 'strikePrice' if 'strikePrice' in options_data.columns else 'strike'
            strike_data = options_data[options_data[strike_col] == strike]
            
            if strike_data.empty:
                return 0.0
            
            # Get CE and PE data - handle different column names
            option_type_col = 'instrumentType' if 'instrumentType' in options_data.columns else 'optionType'
            ce_data = strike_data[strike_data[option_type_col] == 'CE']
            pe_data = strike_data[strike_data[option_type_col] == 'PE']
            
            ce_value = ce_data[metric].sum() if not ce_data.empty else 0
            pe_value = pe_data[metric].sum() if not pe_data.empty else 0
            
            # Calculate PCR
            if ce_value > 0:
                pcr = pe_value / ce_value
            else:
                pcr = 0.0
            
            return pcr
            
        except Exception as e:
            logger.error(f"❌ Error calculating PCR: {e}")
            return 0.0
    
    def _insert_options_data(self, data: pd.DataFrame):
        """Insert options data into database."""
        try:
            if data.empty:
                return
            
            # Insert data using DuckDB's efficient DataFrame insertion with explicit columns
            self.conn.execute("""
                INSERT INTO intraday_options_data (
                    timestamp, index_symbol, strike_price, expiry_date, option_type,
                    last_price, bid_price, ask_price, volume, open_interest, change_in_oi,
                    implied_volatility, delta, gamma, theta, vega, spot_price, atm_strike,
                    pcr_oi, pcr_volume
                ) SELECT 
                    timestamp, index_symbol, strike_price, expiry_date, option_type,
                    last_price, bid_price, ask_price, volume, open_interest, change_in_oi,
                    implied_volatility, delta, gamma, theta, vega, spot_price, atm_strike,
                    pcr_oi, pcr_volume
                FROM data
            """)
            
            logger.info(f"✅ Inserted {len(data)} options records")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert options data: {e}")
    
    def collect_index_data(self, index_symbol: str = 'NIFTY') -> pd.DataFrame:
        """
        Collect 15-minute OHLCV data for the specified index.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            
        Returns:
            DataFrame with index OHLCV data
        """
        try:
            logger.info(f"📈 Collecting index data for {index_symbol}")
            
            # Get current index data using get_index_details
            current_data = self.nse.get_index_details(index_symbol)
            
            if current_data is None or current_data.empty:
                logger.warning(f"❌ No index data available for {index_symbol}")
                return pd.DataFrame()
            
            # Process the data
            processed_data = self._process_index_data(current_data, index_symbol)
            
            # Insert into database
            if not processed_data.empty:
                self._insert_index_data(processed_data)
                logger.info(f"✅ Collected index data for {index_symbol}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting index data for {index_symbol}: {e}")
            return pd.DataFrame()
    
    def _process_index_data(self, index_data: pd.DataFrame, index_symbol: str) -> pd.DataFrame:
        """Process index data into OHLCV format."""
        try:
            current_time = datetime.now()
            
            # Get the first row (main index data)
            if index_data.empty:
                return pd.DataFrame()
            
            # Get the index row (usually first row)
            index_row = index_data.iloc[0]
            
            # Extract OHLCV data - use correct column names from NSE API
            open_price = index_row.get('open', 0)
            high_price = index_row.get('dayHigh', 0)
            low_price = index_row.get('dayLow', 0)
            close_price = index_row.get('lastPrice', 0)
            volume = index_row.get('totalTradedVolume', 0)
            turnover = index_row.get('totalTradedValue', 0)
            
            processed_data = pd.DataFrame([{
                'timestamp': current_time,
                'index_symbol': index_symbol,
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'close_price': close_price,
                'volume': volume,
                'turnover': turnover
            }])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing index data: {e}")
            return pd.DataFrame()
    
    def _insert_index_data(self, data: pd.DataFrame):
        """Insert index data into database."""
        try:
            if data.empty:
                return
            
            # Insert data using DuckDB's efficient DataFrame insertion with explicit columns
            self.conn.execute("""
                INSERT INTO intraday_index_data (
                    timestamp, index_symbol, open_price, high_price, low_price, 
                    close_price, volume, turnover
                ) SELECT 
                    timestamp, index_symbol, open_price, high_price, low_price, 
                    close_price, volume, turnover
                FROM data
            """)
            
            logger.info(f"✅ Inserted {len(data)} index records")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert index data: {e}")
    
    def collect_fii_dii_data(self) -> pd.DataFrame:
        """
        Collect FII/DII data for the current day.
        
        Returns:
            DataFrame with FII/DII data
        """
        try:
            logger.info("💰 Collecting FII/DII data")
            
            # Get FII/DII data from NSE utility (using fii_dii_activity method)
            fii_dii_data = self.nse.fii_dii_activity()
            
            if fii_dii_data is None or fii_dii_data.empty:
                logger.warning("❌ No FII/DII data available")
                return pd.DataFrame()
            
            # Process the data
            processed_data = self._process_fii_dii_data(fii_dii_data)
            
            # Insert into database
            if not processed_data.empty:
                self._insert_fii_dii_data(processed_data)
                logger.info("✅ Collected FII/DII data")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting FII/DII data: {e}")
            return pd.DataFrame()
    
    def _process_fii_dii_data(self, fii_dii_data: pd.DataFrame) -> pd.DataFrame:
        """Process FII/DII data."""
        try:
            current_date = date.today()
            
            # Extract FII/DII values - handle DataFrame structure
            if isinstance(fii_dii_data, pd.DataFrame) and not fii_dii_data.empty:
                # Get the first row if it's a DataFrame
                row = fii_dii_data.iloc[0]
                fii_buy = row.get('fii_buy', row.get('FII_BUY', 0))
                fii_sell = row.get('fii_sell', row.get('FII_SELL', 0))
                dii_buy = row.get('dii_buy', row.get('DII_BUY', 0))
                dii_sell = row.get('dii_sell', row.get('DII_SELL', 0))
            else:
                # Handle as dictionary
                fii_buy = fii_dii_data.get('fii_buy', 0)
                fii_sell = fii_dii_data.get('fii_sell', 0)
                dii_buy = fii_dii_data.get('dii_buy', 0)
                dii_sell = fii_dii_data.get('dii_sell', 0)
            
            fii_net = fii_buy - fii_sell
            dii_net = dii_buy - dii_sell
            
            processed_data = pd.DataFrame([{
                'date': current_date,
                'fii_buy': fii_buy,
                'fii_sell': fii_sell,
                'fii_net': fii_net,
                'dii_buy': dii_buy,
                'dii_sell': dii_sell,
                'dii_net': dii_net
            }])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing FII/DII data: {e}")
            return pd.DataFrame()
    
    def _insert_fii_dii_data(self, data: pd.DataFrame):
        """Insert FII/DII data into database."""
        try:
            if data.empty:
                return
            
            # Check if data for today already exists
            current_date = data.iloc[0]['date']
            existing = self.conn.execute(f"SELECT COUNT(*) FROM intraday_fii_dii_data WHERE date = '{current_date}'").fetchone()
            
            if existing[0] > 0:
                # Update existing record
                row = data.iloc[0]
                self.conn.execute(f"""
                    UPDATE intraday_fii_dii_data SET
                        fii_buy = {row['fii_buy']},
                        fii_sell = {row['fii_sell']},
                        fii_net = {row['fii_net']},
                        dii_buy = {row['dii_buy']},
                        dii_sell = {row['dii_sell']},
                        dii_net = {row['dii_net']}
                    WHERE date = '{current_date}'
                """)
                logger.info(f"✅ Updated FII/DII record for {current_date}")
            else:
                # Insert new record
                self.conn.execute("""
                    INSERT INTO intraday_fii_dii_data (
                        date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
                    ) SELECT 
                        date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
                    FROM data
                """)
                logger.info(f"✅ Inserted {len(data)} FII/DII records")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert FII/DII data: {e}")
    
    def collect_vix_data(self) -> pd.DataFrame:
        """
        Collect India VIX data.
        
        Returns:
            DataFrame with VIX data
        """
        try:
            logger.info("📊 Collecting India VIX data")
            
            # Get VIX data from NSE utility using the new method
            vix_data = self.nse.get_india_vix_data()
            
            if vix_data is None:
                logger.warning("❌ No VIX data available")
                return pd.DataFrame()
            
            # Process the data
            processed_data = self._process_vix_data(vix_data)
            
            # Insert into database
            if not processed_data.empty:
                self._insert_vix_data(processed_data)
                logger.info("✅ Collected VIX data")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting VIX data: {e}")
            return pd.DataFrame()
    
    def _process_vix_data(self, vix_data: Dict) -> pd.DataFrame:
        """Process VIX data."""
        try:
            current_time = datetime.now()
            
            # Extract VIX values from the new data structure
            vix_value = vix_data.get('vix', 0)
            vix_change = vix_data.get('change', 0)
            
            # Log the source of VIX data
            source = vix_data.get('source', 'Unknown')
            logger.info(f"📊 VIX Data Source: {source}")
            logger.info(f"📊 VIX Value: {vix_value}, Change: {vix_change}")
            
            processed_data = pd.DataFrame([{
                'timestamp': current_time,
                'vix_value': vix_value,
                'vix_change': vix_change
            }])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing VIX data: {e}")
            return pd.DataFrame()
    
    def _insert_vix_data(self, data: pd.DataFrame):
        """Insert VIX data into database."""
        try:
            if data.empty:
                return
            
            # Insert data using DuckDB's efficient DataFrame insertion
            self.conn.execute("INSERT INTO intraday_vix_data (timestamp, vix_value, vix_change) SELECT timestamp, vix_value, vix_change FROM data")
            
            logger.info(f"✅ Inserted {len(data)} VIX records")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert VIX data: {e}")
    
    def collect_labels_data(self, index_symbol: str = 'NIFTY') -> pd.DataFrame:
        """
        Collect labels data for 15-minute prediction.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            
        Returns:
            DataFrame with labels data
        """
        try:
            logger.info(f"🏷️ Collecting labels data for {index_symbol}")
            
            # Get current index data
            current_data = self.nse.get_index_details(index_symbol)
            
            if current_data is None or current_data.empty:
                logger.warning(f"❌ No index data available for {index_symbol} labels")
                return pd.DataFrame()
            
            # Process the data
            processed_data = self._process_labels_data(current_data, index_symbol)
            
            # Insert into database
            if not processed_data.empty:
                self._insert_labels_data(processed_data)
                logger.info(f"✅ Collected labels data for {index_symbol}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting labels data for {index_symbol}: {e}")
            return pd.DataFrame()
    
    def _process_labels_data(self, index_data: pd.DataFrame, index_symbol: str) -> pd.DataFrame:
        """Process labels data for 15-minute prediction."""
        try:
            current_time = datetime.now()
            
            # Get the index row
            if index_data.empty:
                return pd.DataFrame()
            
            index_row = index_data.iloc[0]
            current_close = index_row.get('lastPrice', index_row.get('CLOSE', 0))
            
            # For now, we'll create a placeholder label
            # In a real implementation, you would need to:
            # 1. Get the future price (15 minutes later)
            # 2. Calculate the direction: UP = +1, DOWN = -1
            # 3. Calculate return percentage
            
            # Placeholder values (in real implementation, these would be calculated)
            future_close = current_close  # Placeholder
            label = 1  # Placeholder (1 for UP, -1 for DOWN)
            return_pct = 0.0  # Placeholder return percentage
            
            processed_data = pd.DataFrame([{
                'timestamp': current_time,
                'index_symbol': index_symbol,
                'label': label,
                'future_close': future_close,
                'current_close': current_close,
                'return_pct': return_pct
            }])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing labels data: {e}")
            return pd.DataFrame()
    
    def _insert_labels_data(self, data: pd.DataFrame):
        """Insert labels data into database."""
        try:
            if data.empty:
                return
            
            # Insert data using DuckDB's efficient DataFrame insertion
            self.conn.execute("INSERT INTO intraday_labels (timestamp, index_symbol, label, future_close, current_close, return_pct) SELECT timestamp, index_symbol, label, future_close, current_close, return_pct FROM data")
            
            logger.info(f"✅ Inserted {len(data)} labels records")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert labels data: {e}")
    
    def collect_all_data(self, index_symbols: List[str] = ['NIFTY', 'BANKNIFTY']) -> Dict[str, pd.DataFrame]:
        """
        Collect all required data for intraday ML.
        
        Args:
            index_symbols: List of index symbols to collect data for
            
        Returns:
            Dictionary with collected data
        """
        try:
            logger.info("🚀 Starting comprehensive data collection")
            
            collected_data = {}
            
            # Collect data for each index
            for index_symbol in index_symbols:
                logger.info(f"📊 Collecting data for {index_symbol}")
                
                # Collect options chain data
                options_data = self.collect_options_chain_data(index_symbol)
                collected_data[f'{index_symbol}_options'] = options_data
                
                # Collect index data (use correct symbol for index details)
                index_symbol_for_details = 'NIFTY 50' if index_symbol == 'NIFTY' else 'NIFTY BANK'
                index_data = self.collect_index_data(index_symbol_for_details)
                collected_data[f'{index_symbol}_index'] = index_data
                
                # Collect labels data (for 15-min prediction)
                labels_data = self.collect_labels_data(index_symbol_for_details)
                collected_data[f'{index_symbol}_labels'] = labels_data
                
                # Small delay to avoid overwhelming the API
                time.sleep(1)
            
            # Collect FII/DII data (once per day)
            fii_dii_data = self.collect_fii_dii_data()
            collected_data['fii_dii'] = fii_dii_data
            
            # Collect VIX data
            vix_data = self.collect_vix_data()
            collected_data['vix'] = vix_data
            
            logger.info("✅ Comprehensive data collection completed")
            
            return collected_data
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive data collection: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("🔒 Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
