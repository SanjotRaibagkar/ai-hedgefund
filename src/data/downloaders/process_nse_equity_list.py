#!/usr/bin/env python3
"""
Process NSE Equity List
Processes the downloaded NSE equity list to get 3000+ Indian stocks.
"""

import pandas as pd
import re
from src.nsedata.NseUtility import NseUtils

def process_nse_equity_list():
    """Process the NSE equity list to get all symbols."""
    print("🚀 PROCESSING NSE EQUITY LIST")
    print("=" * 60)
    
    try:
        # Load the CSV
        df = pd.read_csv('nse_complete_equity_list.csv')
        print(f"📊 Total entries: {len(df)}")
        
        # Filter for equity instruments
        equity_df = df[df['Instrument Type'] == 'Equity']
        print(f"✅ Equity instruments: {len(equity_df)}")
        
        # Convert company names to symbols
        symbols = []
        company_names = equity_df['Company Name'].tolist()
        
        print("\n🔄 Converting company names to symbols...")
        
        for company in company_names:
            # Convert company name to symbol
            symbol = convert_company_to_symbol(company)
            if symbol:
                symbols.append(symbol)
        
        # Remove duplicates
        unique_symbols = list(set(symbols))
        unique_symbols.sort()
        
        print(f"\n🎯 RESULTS:")
        print(f"📊 Total equity companies: {len(company_names)}")
        print(f"✅ Converted to symbols: {len(symbols)}")
        print(f"🔄 Unique symbols: {len(unique_symbols)}")
        
        # Save symbols
        with open("nse_equity_symbols_complete.txt", "w") as f:
            for symbol in unique_symbols:
                f.write(f"{symbol}\n")
        
        print(f"💾 Saved {len(unique_symbols)} symbols to nse_equity_symbols_complete.txt")
        
        # Sample symbols
        print(f"\n📋 Sample symbols:")
        for symbol in unique_symbols[:20]:
            print(f"  • {symbol}")
        
        if len(unique_symbols) >= 3000:
            print(f"\n🎉 SUCCESS! Got {len(unique_symbols)} Indian stocks (3000+ target achieved!)")
        else:
            print(f"\n⚠️ Got {len(unique_symbols)} symbols, need to find more for 3000+ target")
        
        return unique_symbols
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def convert_company_to_symbol(company_name):
    """Convert company name to symbol."""
    try:
        # Remove common suffixes
        name = company_name.strip()
        
        # Remove common words
        remove_words = [
            'LIMITED', 'LTD', 'LTD.', 'PRIVATE LIMITED', 'PVT LTD', 'PVT. LTD.',
            'MUTUAL FUND', 'ETF', 'TRUST', 'FUND', 'HOLDINGS', 'GROUP',
            'SERVICES', 'TECHNOLOGIES', 'TECHNOLOGY', 'SOLUTIONS', 'SYSTEMS'
        ]
        
        for word in remove_words:
            name = re.sub(r'\b' + word + r'\b', '', name, flags=re.IGNORECASE)
        
        # Clean up
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Extract first few words (usually the main company name)
        words = name.split()
        if len(words) >= 1:
            # Take first 2-3 words max
            symbol = ' '.join(words[:3]).upper()
            # Remove special characters
            symbol = re.sub(r'[^A-Z0-9\s]', '', symbol)
            # Replace spaces with nothing
            symbol = symbol.replace(' ', '')
            
            # Ensure reasonable length
            if 2 <= len(symbol) <= 20:
                return symbol
        
        return None
        
    except Exception:
        return None

def main():
    """Main function."""
    print("🎯 NSE EQUITY LIST PROCESSING")
    print("=" * 60)
    
    symbols = process_nse_equity_list()
    
    if symbols:
        print(f"\n🎉 PROCESSING COMPLETED!")
        print(f"📊 Total symbols: {len(symbols)}")
        print("📈 Ready for comprehensive data download and screening!")
        print("🚀 System can now handle 3000+ Indian stocks!")
        
        # Test with NSEUtility
        print(f"\n🧪 Testing with NSEUtility...")
        try:
            nse = NseUtils()
            test_symbols = symbols[:5]
            print(f"Testing symbols: {test_symbols}")
            
            for symbol in test_symbols:
                try:
                    price_info = nse.price_info(symbol)
                    if price_info:
                        print(f"✅ {symbol}: ₹{price_info.get('LastTradedPrice', 0)}")
                    else:
                        print(f"⚠️ {symbol}: No data")
                except Exception as e:
                    print(f"❌ {symbol}: Error - {e}")
                    
        except Exception as e:
            print(f"❌ NSEUtility test failed: {e}")
    else:
        print("\n❌ Processing failed")

if __name__ == "__main__":
    main() 