#!/usr/bin/env python3
"""
Validate NSE Symbols
Test which symbols from our collection have data available using NSEUtility.
"""

import pandas as pd
import json
import time
import logging
from datetime import datetime
from src.nsedata.NseUtility import NseUtils
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('symbol_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NSESymbolValidator:
    """Validate NSE symbols to check data availability."""
    
    def __init__(self, max_workers: int = 10):
        """Initialize the validator."""
        self.nse = NseUtils()
        self.max_workers = max_workers
        self.results = {
            'valid_symbols': [],
            'invalid_symbols': [],
            'validation_time': None,
            'total_tested': 0
        }
        
    def validate_single_symbol(self, symbol: str) -> Tuple[str, bool, str]:
        """Validate a single symbol."""
        try:
            # Try to get price info for the symbol
            price_data = self.nse.price_info(symbol)
            
            if price_data and isinstance(price_data, dict):
                # Check if we have meaningful data
                if (price_data.get('LastTradedPrice', 0) > 0 or 
                    price_data.get('Open', 0) > 0 or
                    price_data.get('High', 0) > 0 or
                    price_data.get('Low', 0) > 0 or
                    price_data.get('Close', 0) > 0):
                    return symbol, True, "Data available"
                else:
                    return symbol, False, "No price data"
            else:
                return symbol, False, "No response"
                
        except Exception as e:
            error_msg = str(e)
            if "No data available" in error_msg:
                return symbol, False, "No data available"
            elif "symbol not found" in error_msg.lower():
                return symbol, False, "Symbol not found"
            else:
                return symbol, False, f"Error: {error_msg}"
    
    def validate_symbols_batch(self, symbols: List[str], batch_size: int = 100) -> Dict:
        """Validate symbols in batches."""
        logger.info(f"Starting validation of {len(symbols)} symbols...")
        
        valid_symbols = []
        invalid_symbols = []
        total_tested = 0
        
        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Validating batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size} ({len(batch)} symbols)")
            
            batch_results = self._validate_batch(batch)
            
            valid_symbols.extend(batch_results['valid'])
            invalid_symbols.extend(batch_results['invalid'])
            total_tested += len(batch)
            
            # Progress update
            logger.info(f"Progress: {total_tested}/{len(symbols)} symbols tested")
            logger.info(f"Valid: {len(valid_symbols)}, Invalid: {len(invalid_symbols)}")
            
            # Save intermediate results
            self._save_intermediate_results(valid_symbols, invalid_symbols, total_tested)
            
            # Rate limiting between batches
            time.sleep(1)
        
        return {
            'valid_symbols': valid_symbols,
            'invalid_symbols': invalid_symbols,
            'total_tested': total_tested
        }
    
    def _validate_batch(self, symbols: List[str]) -> Dict:
        """Validate a batch of symbols using ThreadPoolExecutor."""
        valid_results = []
        invalid_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all validation tasks
            future_to_symbol = {
                executor.submit(self.validate_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol, is_valid, message = future.result()
                
                if is_valid:
                    valid_results.append({
                        'symbol': symbol,
                        'status': 'valid',
                        'message': message
                    })
                else:
                    invalid_results.append({
                        'symbol': symbol,
                        'status': 'invalid',
                        'message': message
                    })
                
                # Rate limiting
                time.sleep(0.1)
        
        return {
            'valid': valid_results,
            'invalid': invalid_results
        }
    
    def _save_intermediate_results(self, valid_symbols: List, invalid_symbols: List, total_tested: int):
        """Save intermediate results to file."""
        results = {
            'valid_symbols': valid_symbols,
            'invalid_symbols': invalid_symbols,
            'total_tested': total_tested,
            'validation_time': datetime.now().isoformat()
        }
        
        with open('symbol_validation_intermediate.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_symbols_from_file(self, file_path: str = 'all_nse_symbols.csv') -> List[str]:
        """Load symbols from CSV file."""
        try:
            df = pd.read_csv(file_path)
            symbols = df['symbol'].tolist()
            logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
            return symbols
        except Exception as e:
            logger.error(f"Error loading symbols from {file_path}: {e}")
            return []
    
    def run_validation(self, symbols: List[str] = None, batch_size: int = 100) -> Dict:
        """Run the complete validation process."""
        start_time = datetime.now()
        
        # Load symbols if not provided
        if symbols is None:
            symbols = self.load_symbols_from_file()
        
        if not symbols:
            logger.error("No symbols to validate!")
            return {}
        
        logger.info(f"Starting validation of {len(symbols)} symbols...")
        
        # Run validation
        results = self.validate_symbols_batch(symbols, batch_size)
        
        # Add metadata
        results['validation_time'] = start_time.isoformat()
        results['completion_time'] = datetime.now().isoformat()
        results['total_symbols'] = len(symbols)
        results['success_rate'] = len(results['valid_symbols']) / len(symbols) * 100
        
        # Save final results
        self._save_final_results(results)
        
        return results
    
    def _save_final_results(self, results: Dict):
        """Save final validation results."""
        # Save as JSON
        with open('symbol_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save valid symbols as CSV
        if results['valid_symbols']:
            valid_df = pd.DataFrame(results['valid_symbols'])
            valid_df.to_csv('valid_nse_symbols.csv', index=False)
        
        # Save invalid symbols as CSV
        if results['invalid_symbols']:
            invalid_df = pd.DataFrame(results['invalid_symbols'])
            invalid_df.to_csv('invalid_nse_symbols.csv', index=False)
        
        logger.info("Validation results saved to files")

def main():
    """Main function to run symbol validation."""
    print("🔍 NSE SYMBOL VALIDATOR")
    print("=" * 50)
    
    # Initialize validator
    validator = NSESymbolValidator(max_workers=5)  # Conservative rate limiting
    
    try:
        # Run validation
        results = validator.run_validation(batch_size=50)
        
        if not results:
            print("❌ No results obtained from validation")
            return
        
        # Display results
        print("\n📊 VALIDATION RESULTS:")
        print("=" * 30)
        print(f"✅ Valid Symbols: {len(results['valid_symbols'])}")
        print(f"❌ Invalid Symbols: {len(results['invalid_symbols'])}")
        print(f"📋 Total Tested: {results['total_tested']}")
        print(f"📈 Success Rate: {results['success_rate']:.2f}%")
        print(f"🕐 Validation Time: {results['validation_time']}")
        print(f"🕐 Completion Time: {results['completion_time']}")
        
        # Show sample valid symbols
        if results['valid_symbols']:
            print(f"\n📋 Sample Valid Symbols (first 20):")
            for item in results['valid_symbols'][:20]:
                print(f"   • {item['symbol']}")
            
            if len(results['valid_symbols']) > 20:
                print(f"   ... and {len(results['valid_symbols']) - 20} more")
        
        # Show sample invalid symbols
        if results['invalid_symbols']:
            print(f"\n📋 Sample Invalid Symbols (first 10):")
            for item in results['invalid_symbols'][:10]:
                print(f"   • {item['symbol']} - {item['message']}")
            
            if len(results['invalid_symbols']) > 10:
                print(f"   ... and {len(results['invalid_symbols']) - 10} more")
        
        print(f"\n📄 Files Created:")
        print(f"   • symbol_validation_results.json")
        print(f"   • valid_nse_symbols.csv")
        print(f"   • invalid_nse_symbols.csv")
        print(f"   • symbol_validation.log")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        logger.error(f"Validation failed: {e}")

if __name__ == "__main__":
    main()

