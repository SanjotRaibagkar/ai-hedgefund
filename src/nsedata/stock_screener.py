# '''
# Stock Screener

# This script screens stocks based on a set of technical criteria.

# Criteria:
# 1. Today's volume is at least double the 20-day simple moving average of volume.
# 2. Today's closing price is a new 52-week and 90-day high.
# 3. The 14-day Average True Range (ATR) as a percentage of the closing price is greater than 3%.
# 4. The average daily turnover is greater than 1 crore (1e7).

#'''
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from NSEMasterData import NSEMasterData
from NseUtility import NseUtils

def calculate_atr(df, period=14):
    '''Calculates the Average True Range (ATR) for a given DataFrame.'''
    df['high-low'] = df['High'] - df['Low']
    df['high-close'] = abs(df['High'] - df['Close'].shift())
    df['low-close'] = abs(df['Low'] - df['Close'].shift())
    df['tr'] = df[['high-low', 'high-close', 'low-close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr'].iloc[-1]

def process_stock(stock, nse_master, start_date, today):
    '''Processes a single stock to check if it meets the screening criteria.'''
    try:
        print(f"Processing {stock}...")
        hist_data = nse_master.get_history(symbol=stock, exchange='NSE', start=start_date, end=today, interval='1d')

        if hist_data.empty or len(hist_data) < 90:
            print(f"Not enough historical data for {stock}. Skipping.")
            return None

        volume_today = hist_data['Volume'].iloc[-1]
        close_today = hist_data['Close'].iloc[-1]
        sma_volume_20 = hist_data['Volume'].rolling(window=20).mean().iloc[-1]
        high_52wk = hist_data['High'].max()
        high_last_90d = hist_data['High'].tail(90).max()
        atr_14 = calculate_atr(hist_data.copy())
        hist_data['turnover'] = hist_data['Close'] * hist_data['Volume']
        avg_daily_turnover = hist_data['turnover'].tail(20).mean()

        if (
            (volume_today >= 2 * sma_volume_20) and \
            (close_today > max(high_52wk, high_last_90d)) and \
            ((atr_14 / close_today) > 0.03) and \
            (avg_daily_turnover > 1e7)
        ):
            print(f"{stock} shortlisted!")
            return stock
        else:
            print(f"{stock} did not meet the criteria.")
            return None

    except Exception as e:
        print(f"An error occurred while processing {stock}: {e}")
        return None

def screen_stocks():
    '''Screens stocks based on the specified criteria using parallel processing.'''
    nse_master = NSEMasterData()
    nse_master.download_symbol_master()
    nse_utility = NseUtils()
    #stock_universe = nse_utility.get_fno_full_list(list_only=True)
    stock_universe = nse_utility.get_equity_full_list(list_only=True)
    shortlisted = []
    today = datetime.now()
    start_date = today - timedelta(days=365)

    print("Screening stocks...")

    with ThreadPoolExecutor(max_workers=10) as executor: # Adjust max_workers as needed
        future_to_stock = {executor.submit(process_stock, stock, nse_master, start_date, today): stock for stock in stock_universe}
        for future in as_completed(future_to_stock):
            result = future.result()
            if result:
                shortlisted.append(result)

    return shortlisted

if __name__ == "__main__":
    shortlisted_stocks = screen_stocks()
    if shortlisted_stocks:
        print("\n--- Shortlisted Stocks ---")
        for stock_symbol in shortlisted_stocks:
            print(stock_symbol)
    else:
        print("\nNo stocks met the screening criteria.")