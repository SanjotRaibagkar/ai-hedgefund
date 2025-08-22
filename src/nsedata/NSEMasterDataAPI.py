from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from NSEMasterData import NSEMasterData
import pandas as pd

app = Flask(__name__)
nse = NSEMasterData()
nse.download_symbol_master()  # Download master data at startup

@app.route('/history', methods=['GET'])
def get_history():
    symbol = request.args.get('symbol')
    exchange = request.args.get('exchange', 'NSE')
    interval = request.args.get('interval', '1d')
    days = int(request.args.get('days', 6))

    if not symbol:
        return jsonify({'error': 'symbol parameter is required'}), 400

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    data = nse.get_history(
        symbol=symbol,
        exchange=exchange,
        start=start_date,
        end=end_date,
        interval=interval
    )

    if data.empty:
        return jsonify({'error': 'No data found'}), 404

    # Ensure the index is a datetime and reset it
    if not data.index.name or 'date' not in data.columns:
        data = data.reset_index()
    if 'date' in data.columns:
        data['timestamp'] = pd.to_datetime(data['date'])
    elif data.index.name:
        data['timestamp'] = pd.to_datetime(data[data.index.name])
    else:
        data['timestamp'] = pd.to_datetime(data.index)

    # Select only relevant columns for candlestick data
    candle_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    available_cols = [col for col in candle_cols if col in data.columns]
    candles = data[available_cols]

    # Convert timestamp to ISO format
    candles['timestamp'] = candles['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    return jsonify(candles.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

    # to run the rest api on browser use http://127.0.0.1:5000/history?symbol=NIFTY&exchange=NSE&interval=1d&days=6