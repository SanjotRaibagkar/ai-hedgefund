import pandas as pd
from datetime import datetime, timedelta
from NseUtility import NseUtils
dateformat="%d-%m-%Y"
def get_positional_index_data(index: str, interval: str, limit: int, daybefore: int):
    nse = NseUtils()
    today = datetime.today() - timedelta(days=daybefore)

    if interval == '1d':
        from_date = (today - timedelta(days=limit + 150)).strftime(dateformat) # Fetch extra to account for holidays
        df = nse.get_index_historic_data(index, from_date, today.strftime(dateformat))
        df = df.tail(limit).reset_index(drop=True)
        candles = [
            [
                pd.to_datetime(row['TIMESTAMP'], dayfirst=True).strftime('%Y-%m-%d'),
                str(row['OPEN_INDEX_VAL']),
                str(row['HIGH_INDEX_VAL']),
                str(row['LOW_INDEX_VAL']),
                str(row['CLOSE_INDEX_VAL']),
                str(row['TRADED_QTY']),
                pd.to_datetime(row['TIMESTAMP'], dayfirst=True).strftime('%Y-%m-%d'),
                str(row['TURN_OVER']),
                0, "0", "0", "0"
            ] for _, row in df.iterrows()
        ]
        return candles

    elif interval == '1w':
        from_date = (today - timedelta(weeks=limit + 20)).strftime(dateformat)
        df = nse.get_index_historic_data(index, from_date, today.strftime(dateformat))
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], dayfirst=True)
        df = df[df['TIMESTAMP'] < today - timedelta(days=today.weekday())]
        df['week_start'] = df['TIMESTAMP'] - pd.to_timedelta(df['TIMESTAMP'].dt.dayofweek, unit='d')
        weekly_df = df.groupby('week_start').agg(
            OPEN_INDEX_VAL=('OPEN_INDEX_VAL', 'first'),
            HIGH_INDEX_VAL=('HIGH_INDEX_VAL', 'max'),
            LOW_INDEX_VAL=('LOW_INDEX_VAL', 'min'),
            CLOSE_INDEX_VAL=('CLOSE_INDEX_VAL', 'last'),
            TRADED_QTY=('TRADED_QTY', 'sum'),
            TURN_OVER=('TURN_OVER', 'sum'),
            INDEX_NAME=('INDEX_NAME', 'first')
        ).tail(limit)
        candles = [
            [
                row.name.strftime('%Y-%m-%d'),
                str(row['OPEN_INDEX_VAL']),
                str(row['HIGH_INDEX_VAL']),
                str(row['LOW_INDEX_VAL']),
                str(row['CLOSE_INDEX_VAL']),
                str(row['TRADED_QTY']),
                row.name.strftime('%Y-%m-%d'),
                str(row['TURN_OVER']),
                0, "0", "0", "0"
            ] for _, row in weekly_df.iterrows()
        ]
        return candles

    elif interval == '1m':
        from_date = (today - timedelta(days=limit * 31 + 60)).strftime(dateformat)
        df = nse.get_index_historic_data(index, from_date, today.strftime(dateformat))
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], dayfirst=True)
        df = df[~((df['TIMESTAMP'].dt.month == today.month) & (df['TIMESTAMP'].dt.year == today.year))]
        df['month_start'] = df['TIMESTAMP'].dt.to_period('M').dt.start_time
        monthly_df = df.groupby('month_start').agg(
            OPEN_INDEX_VAL=('OPEN_INDEX_VAL', 'first'),
            HIGH_INDEX_VAL=('HIGH_INDEX_VAL', 'max'),
            LOW_INDEX_VAL=('LOW_INDEX_VAL', 'min'),
            CLOSE_INDEX_VAL=('CLOSE_INDEX_VAL', 'last'),
            TRADED_QTY=('TRADED_QTY', 'sum'),
            TURN_OVER=('TURN_OVER', 'sum'),
            INDEX_NAME=('INDEX_NAME', 'first')
        ).tail(limit)
        candles = [
            [
                row.name.strftime('%Y-%m-%d'),
                str(row['OPEN_INDEX_VAL']),
                str(row['HIGH_INDEX_VAL']),
                str(row['LOW_INDEX_VAL']),
                str(row['CLOSE_INDEX_VAL']),
                str(row['TRADED_QTY']),
                row.name.strftime('%Y-%m-%d'),
                str(row['TURN_OVER']),
                0, "0", "0", "0"
            ] for _, row in monthly_df.iterrows()
        ]
        return candles

    return []

# Example usage:
if __name__ == "__main__":
    index_name = "NIFTY 50"
    data = get_positional_index_data(index_name, '1w', 3, 10)
    import json
    print(json.dumps(data, indent=2, default=str))