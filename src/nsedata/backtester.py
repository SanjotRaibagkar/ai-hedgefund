
# Stock Screener Backtester

# This script backtests the screening criteria from the stock_screener.py file
# for a specified number of days.

# It uses parallel processing to speed up the backtesting of all F&O stocks.
# The final output is a table summarizing which stocks would have been
# shortlisted on each of the backtested days.

import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from NSEMasterData import NSEMasterData
from NseUtility import NseUtils

def calculate_atr(df, period=14):
    '''Calculates the Average True Range (ATR) for a given DataFrame.'''
    df_atr = df.copy()
    df_atr['high-low'] = df_atr['High'] - df_atr['Low']
    df_atr['high-close'] = abs(df_atr['High'] - df_atr['Close'].shift())
    df_atr['low-close'] = abs(df_atr['Low'] - df_atr['Close'].shift())
    df_atr['tr'] = df_atr[['high-low', 'high-close', 'low-close']].max(axis=1)
    df_atr['atr'] = df_atr['tr'].rolling(window=period).mean()
    return df_atr['atr'].iloc[-1]

def process_stock_for_backtest(stock, nse_master, end_date, days_to_backtest):
    '''Processes a single stock for the entire backtesting period.'''
    try:
        # Fetch data for the entire period needed (365 days for lookback + backtest duration)
        start_date = end_date - timedelta(days=365 + days_to_backtest)
        hist_data_full = nse_master.get_history(symbol=stock, exchange='NSE', start=start_date, end=end_date, interval='1d')
        
        if hist_data_full.empty or len(hist_data_full) < 252:
            return (stock, [])

        shortlisted_dates = []

        for i in range(days_to_backtest):
            current_date = end_date - timedelta(days=i)
           # print(f"Processing {stock}...date range {start_date}.. ")
            # Create a view of the dataframe up to the current backtesting day
            hist_data = hist_data_full[hist_data_full.index <= current_date].copy()

            if len(hist_data) < 90:
                continue

            # --- Apply Screening Conditions for the current day ---
            volume_today = hist_data['Volume'].iloc[-1]
            close_today = hist_data['Close'].iloc[-1]
            sma_volume_20 = hist_data['Volume'].rolling(window=10).mean().iloc[-1]
            high_52wk = hist_data['High'].tail(252).max() # Using 252 trading days
            high_last_90d = hist_data['High'].tail(20).max()
            atr_14 = calculate_atr(hist_data, period=14)
            hist_data['turnover'] = hist_data['Close'] * hist_data['Volume']
            avg_daily_turnover = hist_data['turnover'].tail(20).mean()

            if (
                (volume_today >= 1.5 * sma_volume_20) and \
                (close_today > high_last_90d) # max(high_52wk, high_last_90d)) #and \
                #(atr_14 is not None and (atr_14 / close_today) > 0.03) #and \
                #(avg_daily_turnover > 1e7)
            ):
                shortlisted_dates.append(current_date.strftime('%Y-%m-%d'))
        
        return (stock, shortlisted_dates)

    except Exception:
        # Silently handle errors for individual stocks
        return (stock, [])

def backtest_screener(days_to_backtest=3):
    '''
    Backtests the screener criteria for the given number of days.

    Args:
        days_to_backtest (int): The number of past days to run the backtest on.
    '''
    nse_master = NSEMasterData()
    nse_master.download_symbol_master()
    nse_utility = NseUtils()
    stock_universe = nse_utility.get_fno_full_list(list_only=True)

    backtest_results = {}
    today = datetime.now()

    # --- Print the dates being backtested ---
    backtest_dates = [ (today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_to_backtest) ]
    print(f"Backtesting for the following dates: {', '.join(backtest_dates)}\n")

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create futures for all stocks
        future_to_stock = {
            executor.submit(process_stock_for_backtest, stock, nse_master, today, days_to_backtest): stock 
            for stock in stock_universe
        }

        for future in as_completed(future_to_stock):
            stock, dates = future.result()
            if dates:
                for date_str in dates:
                    if date_str not in backtest_results:
                        backtest_results[date_str] = []
                    backtest_results[date_str].append(stock)

    # --- Format and Print Results ---
    if not backtest_results:
        print("No stocks were shortlisted during the backtest period.")
        return

    # Sort dates descending
    sorted_dates = sorted(backtest_results.keys(), reverse=True)

    results_df = pd.DataFrame([
        {"Date": date, "Shortlisted Stocks": ", ".join(sorted(backtest_results[date]))}
        for date in sorted_dates
    ])

    print("\n--- Backtesting Results ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    # You can change the number of days to backtest here
    # For example, to backtest for the last 5 days: backtest_screener(days_to_backtest=5)
    backtest_screener(days_to_backtest=30)
