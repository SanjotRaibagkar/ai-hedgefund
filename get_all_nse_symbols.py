#!/usr/bin/env python3
"""
Get All NSE Symbols
Comprehensive script to fetch all available NSE symbols using multiple methods.
"""

import pandas as pd
import requests
import os
import time
from datetime import datetime
from src.nsedata.NseUtility import NseUtils
import json
from typing import List, Dict, Set
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NSESymbolCollector:
    """Collect all NSE symbols using multiple methods."""
    
    def __init__(self):
        """Initialize the collector."""
        self.nse = NseUtils()
        self.symbols = set()
        self.output_file = "all_nse_symbols.json"
        
    def get_symbols_from_nse_utility(self) -> Set[str]:
        """Get symbols using NSEUtility methods."""
        logger.info("Fetching symbols using NSEUtility...")
        symbols = set()
        
        try:
            # Method 1: Get all indices
            indices = ['NIFTY 50', 'NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200', 
                      'NIFTY 500', 'NIFTY SMALLCAP 100', 'NIFTY MIDCAP 100',
                      'NIFTY BANK', 'NIFTY AUTO', 'NIFTY FMCG', 'NIFTY IT',
                      'NIFTY MEDIA', 'NIFTY METAL', 'NIFTY PHARMA', 'NIFTY PSU BANK',
                      'NIFTY REALTY', 'NIFTY PRIVATE BANK', 'NIFTY CONSUMER DURABLES',
                      'NIFTY ENERGY', 'NIFTY FINANCIAL SERVICES']
            
            for index in indices:
                try:
                    logger.info(f"Fetching symbols for {index}...")
                    index_data = self.nse.get_index_pe_history(index)
                    if index_data and hasattr(index_data, 'index'):
                        # Extract symbols from index data
                        if hasattr(index_data.index, 'tolist'):
                            index_symbols = index_data.index.tolist()
                            symbols.update(index_symbols)
                            logger.info(f"Added {len(index_symbols)} symbols from {index}")
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error fetching {index}: {e}")
                    continue
            
            # Method 2: Get bhavcopy data
            try:
                logger.info("Fetching bhavcopy data...")
                bhavcopy_data = self.nse.bhavcopy()
                if bhavcopy_data:
                    # Extract symbols from bhavcopy
                    if hasattr(bhavcopy_data, 'columns') and 'SYMBOL' in bhavcopy_data.columns:
                        bhavcopy_symbols = bhavcopy_data['SYMBOL'].dropna().unique().tolist()
                        symbols.update(bhavcopy_symbols)
                        logger.info(f"Added {len(bhavcopy_symbols)} symbols from bhavcopy")
            except Exception as e:
                logger.warning(f"Error fetching bhavcopy: {e}")
            
        except Exception as e:
            logger.error(f"Error in NSEUtility method: {e}")
        
        return symbols
    
    def get_symbols_from_nse_website(self) -> Set[str]:
        """Get symbols by downloading from NSE website."""
        logger.info("Fetching symbols from NSE website...")
        symbols = set()
        
        try:
            # NSE Equity list URL
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                # Save the file
                with open('nse_equity_latest.csv', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Parse CSV
                df = pd.read_csv('nse_equity_latest.csv')
                if 'SYMBOL' in df.columns:
                    equity_symbols = df['SYMBOL'].dropna().unique().tolist()
                    symbols.update(equity_symbols)
                    logger.info(f"Added {len(equity_symbols)} symbols from NSE website")
                
        except Exception as e:
            logger.warning(f"Error fetching from NSE website: {e}")
        
        return symbols
    
    def get_symbols_from_existing_csv(self) -> Set[str]:
        """Get symbols from existing CSV files."""
        logger.info("Reading symbols from existing CSV files...")
        symbols = set()
        
        csv_files = [
            'nse_complete_equity_list.csv',
            'nse_equity_raw.csv'
        ]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Try different possible column names
                    symbol_columns = ['SYMBOL', 'Symbol', 'symbol', 'Company Name', 'COMPANY_NAME']
                    
                    for col in symbol_columns:
                        if col in df.columns:
                            file_symbols = df[col].dropna().unique().tolist()
                            # Clean symbols (remove .NS suffix, etc.)
                            cleaned_symbols = []
                            for symbol in file_symbols:
                                if isinstance(symbol, str):
                                    cleaned = symbol.replace('.NS', '').replace('.NSE', '').strip()
                                    if len(cleaned) > 0:
                                        cleaned_symbols.append(cleaned.upper())
                            
                            symbols.update(cleaned_symbols)
                            logger.info(f"Added {len(cleaned_symbols)} symbols from {csv_file}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Error reading {csv_file}: {e}")
        
        return symbols
    
    def get_symbols_from_indices(self) -> Set[str]:
        """Get symbols from major NSE indices."""
        logger.info("Fetching symbols from major indices...")
        symbols = set()
        
        # Major NSE indices
        indices = [
            'NIFTY 50',
            'NIFTY NEXT 50', 
            'NIFTY 100',
            'NIFTY 200',
            'NIFTY 500',
            'NIFTY SMALLCAP 100',
            'NIFTY MIDCAP 100',
            'NIFTY BANK',
            'NIFTY AUTO',
            'NIFTY FMCG',
            'NIFTY IT',
            'NIFTY MEDIA',
            'NIFTY METAL',
            'NIFTY PHARMA',
            'NIFTY PSU BANK',
            'NIFTY REALTY',
            'NIFTY PRIVATE BANK',
            'NIFTY CONSUMER DURABLES',
            'NIFTY ENERGY',
            'NIFTY FINANCIAL SERVICES'
        ]
        
        for index in indices:
            try:
                logger.info(f"Fetching {index}...")
                # Try different methods to get index constituents
                
                # Method 1: Try get_index_pe_history
                try:
                    index_data = self.nse.get_index_pe_history(index)
                    if index_data is not None:
                        if hasattr(index_data, 'index'):
                            index_symbols = index_data.index.tolist()
                            symbols.update(index_symbols)
                            logger.info(f"Added {len(index_symbols)} symbols from {index}")
                except:
                    pass
                
                # Method 2: Try get_index_pe
                try:
                    index_data = self.nse.get_index_pe(index)
                    if index_data is not None:
                        if hasattr(index_data, 'index'):
                            index_symbols = index_data.index.tolist()
                            symbols.update(index_symbols)
                            logger.info(f"Added {len(index_symbols)} symbols from {index}")
                except:
                    pass
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error fetching {index}: {e}")
                continue
        
        return symbols
    
    def clean_and_validate_symbols(self, symbols: Set[str]) -> Set[str]:
        """Clean and validate symbols."""
        logger.info("Cleaning and validating symbols...")
        cleaned_symbols = set()
        
        for symbol in symbols:
            if isinstance(symbol, str):
                # Clean the symbol
                cleaned = symbol.strip().upper()
                cleaned = cleaned.replace('.NS', '').replace('.NSE', '')
                cleaned = cleaned.replace(' ', '').replace('-', '')
                
                # Basic validation
                if (len(cleaned) >= 2 and 
                    len(cleaned) <= 20 and 
                    cleaned.isalnum() and
                    not cleaned.isdigit()):
                    cleaned_symbols.add(cleaned)
        
        logger.info(f"Cleaned {len(symbols)} symbols to {len(cleaned_symbols)} valid symbols")
        return cleaned_symbols
    
    def collect_all_symbols(self) -> Dict[str, List[str]]:
        """Collect all symbols using all available methods."""
        logger.info("Starting comprehensive symbol collection...")
        
        all_symbols = set()
        method_results = {}
        
        # Method 1: NSEUtility
        try:
            nse_symbols = self.get_symbols_from_nse_utility()
            all_symbols.update(nse_symbols)
            method_results['nse_utility'] = list(nse_symbols)
            logger.info(f"NSEUtility method: {len(nse_symbols)} symbols")
        except Exception as e:
            logger.error(f"Error in NSEUtility method: {e}")
            method_results['nse_utility'] = []
        
        # Method 2: NSE Website
        try:
            website_symbols = self.get_symbols_from_nse_website()
            all_symbols.update(website_symbols)
            method_results['nse_website'] = list(website_symbols)
            logger.info(f"NSE Website method: {len(website_symbols)} symbols")
        except Exception as e:
            logger.error(f"Error in NSE Website method: {e}")
            method_results['nse_website'] = []
        
        # Method 3: Existing CSV files
        try:
            csv_symbols = self.get_symbols_from_existing_csv()
            all_symbols.update(csv_symbols)
            method_results['existing_csv'] = list(csv_symbols)
            logger.info(f"Existing CSV method: {len(csv_symbols)} symbols")
        except Exception as e:
            logger.error(f"Error in Existing CSV method: {e}")
            method_results['existing_csv'] = []
        
        # Method 4: Indices
        try:
            index_symbols = self.get_symbols_from_indices()
            all_symbols.update(index_symbols)
            method_results['indices'] = list(index_symbols)
            logger.info(f"Indices method: {len(index_symbols)} symbols")
        except Exception as e:
            logger.error(f"Error in Indices method: {e}")
            method_results['indices'] = []
        
        # Clean and validate all symbols
        cleaned_symbols = self.clean_and_validate_symbols(all_symbols)
        
        # Prepare final results
        results = {
            'total_symbols': len(cleaned_symbols),
            'collection_time': datetime.now().isoformat(),
            'methods_used': method_results,
            'all_symbols': sorted(list(cleaned_symbols))
        }
        
        # Save results
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as CSV
        df = pd.DataFrame({'symbol': sorted(list(cleaned_symbols))})
        df.to_csv('all_nse_symbols.csv', index=False)
        
        logger.info(f"Collection complete! Total unique symbols: {len(cleaned_symbols)}")
        logger.info(f"Results saved to {self.output_file} and all_nse_symbols.csv")
        
        return results

def main():
    """Main function to run the symbol collection."""
    print("🚀 NSE SYMBOL COLLECTOR")
    print("=" * 50)
    
    collector = NSESymbolCollector()
    
    try:
        results = collector.collect_all_symbols()
        
        print("\n📊 COLLECTION RESULTS:")
        print("=" * 30)
        print(f"✅ Total Unique Symbols: {results['total_symbols']}")
        print(f"🕐 Collection Time: {results['collection_time']}")
        
        print("\n📋 Methods Used:")
        for method, symbols in results['methods_used'].items():
            print(f"   • {method}: {len(symbols)} symbols")
        
        print(f"\n📄 Files Created:")
        print(f"   • {collector.output_file}")
        print(f"   • all_nse_symbols.csv")
        
        # Show sample symbols
        if results['all_symbols']:
            print(f"\n📋 Sample Symbols (first 20):")
            for symbol in results['all_symbols'][:20]:
                print(f"   • {symbol}")
            
            if len(results['all_symbols']) > 20:
                print(f"   ... and {len(results['all_symbols']) - 20} more")
        
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        logger.error(f"Collection failed: {e}")

if __name__ == "__main__":
    main()

