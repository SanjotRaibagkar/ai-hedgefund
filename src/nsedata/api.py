from flask import Flask, request, jsonify
from nsepostionaldata import get_positional_index_data

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_data():
    index = request.args.get('index', default='NIFTY 50', type=str)
    interval = request.args.get('interval', default='1d', type=str)
    limit = request.args.get('limit', default=200, type=int)
    daybefore = request.args.get('daybefore', default=0, type=int)

    # Map API interval to a more descriptive timeframe for the output
    timeframe_map = {
        '1d': '1day',
        '1w': '1week',
        '1m': '1month'
    }
    timeframe = timeframe_map.get(interval, '1day')

    data = get_positional_index_data(index, interval, limit, daybefore)
    
    # Structure the final JSON output
    result = [{
        "timeframe": timeframe,
        "candles": data
    }]
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
