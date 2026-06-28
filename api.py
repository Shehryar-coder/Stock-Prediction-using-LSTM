from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from dashboard import (
    load_master, fetch_live, get_price_changes, calculate_rsi, get_signal,
    run_prediction, run_validation, get_top_gainers, get_top_losers
)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load master data once
master_df = load_master()

@app.route('/api/coins', methods=['GET'])
def get_coins():
    coins = []
    for _, row in master_df.iterrows():
        coins.append({
            'id': str(row['Symbol']).lower(),
            'name': str(row['Coin Name']),
            'symbol': str(row['Symbol'])
        })
    return jsonify(coins)

@app.route('/api/stats/<coin_symbol>', methods=['GET'])
def get_stats(coin_symbol):
    try:
        hist_df = fetch_live(coin_symbol.upper())
        if hist_df.empty:
            return jsonify({'error': 'No data available'}), 404

        prices = get_price_changes(hist_df)
        rsi_vals = calculate_rsi(hist_df['Close'])
        latest_rsi = float(rsi_vals.iloc[-1])

        return jsonify({
            'currentPrice': prices['current'],
            'change24h': prices['change_24h'],
            'changeAmount24h': prices['current'] * (prices['change_24h'] / 100),
            'high24h': prices['current'] * 1.02,  # Approximate
            'low24h': prices['current'] * 0.98,   # Approximate
            'rsi': latest_rsi,
            'signal': get_signal(latest_rsi)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/price-history/<coin_symbol>', methods=['GET'])
def get_price_history(coin_symbol):
    try:
        range_param = request.args.get('range', '1D')
        hist_df = fetch_live(coin_symbol.upper())
        if hist_df.empty:
            return jsonify([]), 200

        # Convert to the format expected by frontend
        data = []
        for timestamp, row in hist_df.iterrows():
            data.append({
                'timestamp': timestamp.isoformat(),
                'price': float(row['Close'])
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction/<coin_symbol>', methods=['GET'])
def get_prediction(coin_symbol):
    try:
        hist_df = fetch_live(coin_symbol.upper())
        if hist_df.empty:
            return jsonify({'error': 'No data available'}), 404

        pred_price = run_prediction(hist_df)
        current_price = float(hist_df['Close'].iloc[-1])
        change_pct = ((pred_price - current_price) / current_price) * 100

        # Run validation for accuracy
        val_result = run_validation(hist_df)

        return jsonify({
            'predictedPrice': pred_price,
            'expectedChangePercentage': change_pct,
            'expectedChangeAmount': pred_price - current_price,
            'modelAccuracy': 100 - val_result['mape'],  # Convert MAPE to accuracy
            'deviationPercentage': val_result['mape'],
            'averageAccuracy7d': 100 - val_result['mape']  # Placeholder
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market/<type>', methods=['GET'])
def get_market_data(type):
    try:
        if type == 'gainers':
            df = get_top_gainers(master_df)
        elif type == 'losers':
            df = get_top_losers(master_df)
        else:
            return jsonify({'error': 'Invalid type'}), 400

        data = []
        for _, row in df.iterrows():
            data.append({
                'id': str(row['Symbol']).lower(),
                'name': str(row['Coin Name']),
                'symbol': str(row['Symbol']),
                'price': float(row['Price']),
                'change24h': float(row['24h'])
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/deviation-data/<coin_symbol>', methods=['GET'])
def get_deviation_data(coin_symbol):
    try:
        hist_df = fetch_live(coin_symbol.upper())
        if hist_df.empty:
            return jsonify([]), 200

        # For simplicity, return last 30 days with some mock deviation
        # In a real implementation, this would use actual prediction history
        data = []
        close_prices = hist_df['Close'].tail(30)

        for i, (timestamp, row) in enumerate(close_prices.items()):
            actual = float(row)
            # Mock predicted price with some deviation
            predicted = actual * (1 + (np.random.random() - 0.5) * 0.1)
            deviation_pct = ((predicted - actual) / actual) * 100

            data.append({
                'timestamp': timestamp.isoformat(),
                'actualPrice': actual,
                'predictedPrice': predicted,
                'deviationPercentage': deviation_pct
            })

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
