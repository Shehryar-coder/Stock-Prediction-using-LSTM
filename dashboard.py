# ============================================================
#  dashboard.py  —  Backend v4.0
#
#  ACCURACY IMPROVEMENTS vs v3:
#   1. fetch_live        → start 2017-01-01 (3× more training data)
#   2. fetch_fear_greed  → Fear & Greed Index (alternative.me, free)
#   3. fetch_btc_dominance → BTC dominance proxy (independent signal)
#   4. build_features    → 28 features, KEY FIXES:
#       • Log_Return added  → model predicts CHANGE not level (fixes lag)
#       • BB_PctB added     → where is price in the band?
#       • Price_vs_SMA50/200 → relative position signals
#       • Vol_Ratio instead of raw Vol_SMA
#       • Return_3d/7d/14d/30d → multi-timeframe momentum
#       • HL_Range → intraday range (volatility signal)
#       • Fear_Greed → sentiment (external, independent)
#       • BTC_Dom → market context (external, independent)
#       • CSV_Rank/Mom_24h/7d/30d/VolRank → from CryptocurrencyData.csv
#       • REMOVED: BB_Upper, BB_Lower, EMA_10, MACD, MACD_Signal (redundant)
#   5. make_sequences    → y = next-day Log_Return (not raw Close)
#   6. run_prediction    → converts predicted log_return → price
#      pred_price = last_close × exp(predicted_log_return)
# ============================================================

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from collections import deque
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

USERS_FILE = "users.txt"


# ──────────────────────────────────────────────
#  AUTH
# ──────────────────────────────────────────────
def save_user(username: str, password: str) -> None:
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            f.write("Username|Password\n")
    with open(USERS_FILE, "a") as f:
        f.write(f"{username}|{password}\n")


def user_exists(username: str, password: str) -> bool:
    if not os.path.exists(USERS_FILE):
        return False
    with open(USERS_FILE, "r") as f:
        for line in f.readlines():
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) == 2 and parts[0] == username and parts[1] == password:
                    return True
    return False


def get_all_users() -> set:
    users = set()
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            for line in f.readlines()[1:]:
                if "|" in line:
                    users.add(line.strip().split("|")[0])
    return users


# ──────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────
def clean_val(x):
    if isinstance(x, str):
        clean_str = x.replace('$', '').replace(',', '').replace('%', '').strip()
        if clean_str in ['', '-']:
            return np.nan
        return float(clean_str)
    return x


# ──────────────────────────────────────────────
#  PREDICTION HISTORY
# ──────────────────────────────────────────────
def log_prediction(username: str, coin: str, pred: float, actual: str = "Pending") -> None:
    file_path    = f"{username}_prediction_history.txt"
    display_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    today_str    = date.today().isoformat()
    new_entry = (
        f"{display_date} | {coin} | Prediction: {pred:,.2f} | "
        f"Actual: {actual} | pred_date:{today_str}\n"
    )
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(f"Prediction History of {username}\n")
            f.write("-" * 60 + "\n")
    with open(file_path, "a") as f:
        f.write(new_entry)


def update_actual_prices(username: str) -> None:
    """Only replaces Pending if prediction was made on a previous calendar day."""
    file_path = f"{username}_prediction_history.txt"
    if not os.path.exists(file_path):
        return
    today_str     = date.today().isoformat()
    updated_lines = deque()
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Actual: Pending" in line:
            try:
                pred_date_str = ""
                if "pred_date:" in line:
                    pred_date_str = line.split("pred_date:")[-1].strip()
                if pred_date_str and pred_date_str >= today_str:
                    updated_lines.append(line)
                    continue
                parts  = line.split("|")
                coin   = parts[1].strip()
                ticker = f"{coin}-USD"
                latest = yf.download(ticker, period="2d", progress=False)
                if not latest.empty:
                    actual_price = float(latest['Close'].iloc[-1].squeeze())
                    clean_line   = line.split("| pred_date:")[0].strip()
                    clean_line   = clean_line.replace(
                        "Actual: Pending", f"Actual: {actual_price:,.2f}"
                    ) + "\n"
                    line = clean_line
            except Exception:
                pass
        else:
            if "pred_date:" in line:
                line = line.split("| pred_date:")[0].strip() + "\n"
        updated_lines.append(line)
    with open(file_path, "w") as f:
        f.writelines(updated_lines)


def read_prediction_history(username: str) -> str:
    file_path = f"{username}_prediction_history.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return ""


# ──────────────────────────────────────────────
#  VALIDATE PREDICTION HISTORY — Real-world validator
#  Reads completed predictions from user history file
#  and calculates MAPE based on actual prices
# ──────────────────────────────────────────────
def validate_prediction_history(username: str) -> dict:
    """
    Validates user predictions by reading their prediction history file.
    Calculates MAPE for all completed (non-Pending) predictions.
    
    Returns:
        dict with keys:
        - mape: float or None (overall mean MAPE %)
        - count: int (number of completed predictions)
        - comp_df: pd.DataFrame with columns [Coin, Predicted, Actual, Error, MAPE]
    """
    file_path = f"{username}_prediction_history.txt"

    if not os.path.exists(file_path):
        return {
            "mape": None,
            "count": 0,
            "comp_df": pd.DataFrame()
        }

    rows = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        if "Prediction:" not in line or "Actual:" not in line:
            continue

        if "Pending" in line:
            continue

        try:
            parts = line.split("|")

            coin = parts[1].strip()

            pred_part = [p for p in parts if "Prediction:" in p][0]
            actual_part = [p for p in parts if "Actual:" in p][0]

            pred = float(
                pred_part.split("Prediction:")[1]
                .replace(",", "")
                .strip()
            )

            actual = float(
                actual_part.split("Actual:")[1]
                .replace(",", "")
                .strip()
            )

            error = abs(pred - actual)
            mape = (error / (actual + 1e-10)) * 100

            rows.append({
                "Coin": coin,
                "Predicted": pred,
                "Actual": actual,
                "Error": error,
                "MAPE": mape
            })

        except Exception:
            continue

    if not rows:
        return {
            "mape": None,
            "count": 0,
            "comp_df": pd.DataFrame()
        }

    comp_df = pd.DataFrame(rows)

    overall_mape = comp_df["MAPE"].mean()

    return {
        "mape": overall_mape,
        "count": len(comp_df),
        "comp_df": comp_df
    }


def clear_prediction_history(username: str) -> bool:
    """
    Clears the prediction history file for a given user.
    Returns True if file was deleted, False otherwise.
    """
    file_path = f"{username}_prediction_history.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


# ──────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────
def load_master() -> pd.DataFrame:
    df = pd.read_csv("CryptocurrencyData.csv")
    df.columns = df.columns.str.strip()
    for col in ['Price', '1h', '24h', '7d', '30d', '24h Volume', 'Market Cap']:
        if col in df.columns:
            df[col] = df[col].apply(clean_val)
    return df


def get_csv_context(symbol: str, master_df: pd.DataFrame) -> dict:
    """
    Extract static context features for a coin from CryptocurrencyData.csv.
    Returns normalised rank, momentum %, volume rank.
    Safe — returns neutral values if coin not found.
    """
    row = master_df[master_df['Symbol'].str.strip().str.upper() == symbol.upper()]
    if row.empty:
        return {"rank_norm": 0.5, "chg_24h": 0.0, "chg_7d": 0.0,
                "chg_30d": 0.0, "vol_rank_norm": 0.5}
    r         = row.iloc[0]
    total     = max(len(master_df), 1)
    rank_norm = 1.0 - (float(r.get('Rank', total)) / total)

    vols     = master_df['24h Volume'].dropna()
    coin_vol = float(r.get('24h Volume', 0) or 0)
    vol_rank = float((vols < coin_vol).sum()) / max(len(vols), 1)

    return {
        "rank_norm":     float(np.clip(rank_norm, 0, 1)),
        "chg_24h":       float(r.get('24h',  0) or 0),
        "chg_7d":        float(r.get('7d',   0) or 0),
        "chg_30d":       float(r.get('30d',  0) or 0),
        "vol_rank_norm": float(np.clip(vol_rank, 0, 1)),
    }


# ──────────────────────────────────────────────
#  EXTERNAL SIGNAL: FEAR & GREED INDEX
#  Free — alternative.me — no API key needed.
# ──────────────────────────────────────────────
def fetch_fear_greed(n_days: int = 3000) -> pd.Series:
    """
    Returns pd.Series indexed by date, values 0–1.
    Falls back to empty Series on network failure.
    """
    try:
        url  = f"https://api.alternative.me/fng/?limit={n_days}&format=json"
        resp = requests.get(url, timeout=8)
        data = resp.json().get("data", [])
        if not data:
            return pd.Series(dtype=float)
        dates  = pd.to_datetime([d["timestamp"] for d in data], unit='s')
        values = [float(d["value"]) / 100.0 for d in data]
        return pd.Series(values, index=dates).sort_index()
    except Exception:
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────
#  EXTERNAL SIGNAL: BTC DOMINANCE PROXY
#  BTC / (BTC + ETH supply-weighted) ratio.
#  Aligned to the coin's date index.
# ──────────────────────────────────────────────
def fetch_btc_dominance(index: pd.DatetimeIndex) -> pd.Series:
    """
    Returns pd.Series of BTC dominance proxy (0–1), aligned to index.
    Falls back to 0.45 on error.
    """
    try:
        btc = yf.download("BTC-USD", start="2017-01-01",
                          auto_adjust=True, progress=False)['Close']
        eth = yf.download("ETH-USD", start="2017-01-01",
                          auto_adjust=True, progress=False)['Close']
        if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
        if isinstance(eth, pd.DataFrame): eth = eth.iloc[:, 0]
        btc  = btc.squeeze().reindex(index, method='ffill')
        eth  = eth.squeeze().reindex(index, method='ffill').fillna(1.0)
        dom  = btc / (btc + eth * 120.0)   # rough supply-weighted
        return dom.clip(0, 1).fillna(0.45)
    except Exception:
        return pd.Series(0.45, index=index)


# ──────────────────────────────────────────────
#  FETCH LIVE — start 2017 for max training data
#  Fix: retry without auto_adjust if empty,
#  and handle BTC/ETH MultiIndex columns safely
# ──────────────────────────────────────────────
def fetch_live(coin_symbol: str) -> pd.DataFrame:
    ticker = f"{coin_symbol}-USD"

    # First attempt: with auto_adjust — use more history for training
    data = yf.download(
        ticker,
        start="2014-01-01",
        end=datetime.now(),
        auto_adjust=True,
        progress=False
    )

    # Second attempt without auto_adjust if empty
    if data.empty:
        data = yf.download(
            ticker,
            start="2014-01-01",
            end=datetime.now(),
            auto_adjust=False,
            progress=False
        )

    if data.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns e.g. ('Close','BTC-USD') → 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Select OHLCV — use Adj Close as Close if available
    if 'Adj Close' in data.columns and 'Close' not in data.columns:
        data = data.rename(columns={'Adj Close': 'Close'})

    available = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
    df = data[available].copy()

    # Squeeze any remaining 1-column DataFrames to Series
    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].squeeze()

    # Drop rows where Close is NaN or zero
    df = df[df['Close'].notna() & (df['Close'] > 0)]

    # Forward-fill and drop any remaining NaNs (helps with sparse BTC data)
    df = df.ffill().dropna()

    return df


# ──────────────────────────────────────────────
#  FEATURE ENGINEERING — clean 15 features
#
#  KEY RULES:
#  1. Every feature is already 0-1 or small float
#     BEFORE scaling — no unbounded values
#  2. Close is kept as raw price for target only
#     — all indicator features are normalised
#  3. No raw EMA/SMA prices as features
#     (use price/SMA ratio instead)
#  4. OBV replaced by OBV_pct_change (bounded)
# ──────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   symbol: str = "",
                   master_df: pd.DataFrame = None,
                   fear_greed_series: pd.Series = None,
                   btc_dom_series: pd.Series = None) -> pd.DataFrame:

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    close  = out['Close'].squeeze().astype(float)
    high   = out['High'].squeeze().astype(float)
    low    = out['Low'].squeeze().astype(float)
    volume = out['Volume'].squeeze().astype(float)

    # ── RSI (14) normalised 0-1 ───────────────────────────
    delta  = close.diff()
    gain   = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss   = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi    = (100 - (100 / (1 + gain / (loss + 1e-10)))) / 100.0
    out['RSI'] = rsi.squeeze()

    # ── MACD histogram / ATR normalised ───────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
    atr_denom = close.rolling(14).mean()
    out['MACD_norm'] = (macd_hist / (atr_denom + 1e-10)).squeeze()

    # ── Bollinger %B (0=lower band, 1=upper band) ─────────
    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    out['BB_PctB'] = ((close - bb_lower) /
                      (bb_upper - bb_lower + 1e-10)).clip(0, 1).squeeze()

    # ── Price relative to SMA20/50/200 ────────────────────
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out['P_SMA20']  = (close / (sma20  + 1e-10) - 1.0).squeeze()
    out['P_SMA50']  = (close / (sma50  + 1e-10) - 1.0).squeeze()
    out['P_SMA200'] = (close / (sma200 + 1e-10) - 1.0).squeeze()

    # ── ATR % of price (volatility, normalised) ───────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    out['ATR_pct'] = (tr.rolling(14).mean() /
                      (close + 1e-10)).squeeze()

    # ── Volume ratio (today vs 20-day avg) ────────────────
    vol_ma20 = volume.rolling(20).mean()
    out['Vol_Ratio'] = (volume / (vol_ma20 + 1e-10)).clip(0, 5).squeeze()

    # ── Volume change (log) — important short-term signal
    out['Volume_Change'] = np.log(
        volume / (volume.shift(1) + 1e-10)
    ).clip(-3, 3)

    # ── Log returns (momentum, bounded ~[-0.3, 0.3]) ──────
    out['Ret_1d']  = np.log(close / (close.shift(1)  + 1e-10)).squeeze()
    out['Ret_5d']  = np.log(close / (close.shift(5)  + 1e-10)).squeeze()
    out['Ret_20d'] = np.log(close / (close.shift(20) + 1e-10)).squeeze()

    # ── Williams %R normalised 0-1 ────────────────────────
    hh = high.rolling(14).max()
    ll = low.rolling(14).min()
    out['WilliamsR'] = (1.0 - (hh - close) /
                        (hh - ll + 1e-10)).clip(0, 1).squeeze()

    # ── Fear & Greed (0-1) ────────────────────────────────
    if fear_greed_series is not None and not fear_greed_series.empty:
        fg = fear_greed_series.reindex(out.index, method='ffill').fillna(0.5)
        out['Fear_Greed'] = fg.values
    else:
        out['Fear_Greed'] = 0.5

    # ── BTC Dominance (0-1) ───────────────────────────────
    if btc_dom_series is not None and not btc_dom_series.empty:
        dom = btc_dom_series.reindex(out.index, method='ffill').fillna(0.45)
        out['BTC_Dom'] = dom.values
    else:
        out['BTC_Dom'] = 0.45

    # ── CSV context ───────────────────────────────────────
    if master_df is not None and symbol:
        ctx = get_csv_context(symbol, master_df)
    else:
        ctx = {"rank_norm": 0.5, "chg_24h": 0.0, "chg_7d": 0.0,
               "chg_30d": 0.0, "vol_rank_norm": 0.5}
    out['CSV_Rank']   = ctx["rank_norm"]
    out['CSV_Mom_7d'] = np.clip(ctx["chg_7d"]  / 100.0, -1, 1)
    out['CSV_Mom_30d']= np.clip(ctx["chg_30d"] / 100.0, -1, 1)

    # Keep only feature columns + Close (target)
    feature_cols = [
        'RSI',
        'MACD_norm',
        'BB_PctB',
        'P_SMA20',
        'P_SMA50',
        'ATR_pct',
        'Vol_Ratio',
        'Volume_Change',
        'Ret_1d',
        'Ret_5d',
        'Ret_20d',
        'WilliamsR',
        'Fear_Greed',
        'BTC_Dom',
        'CSV_Rank',
    ]
    out = out[feature_cols + ['Close']].copy()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    return out


# ──────────────────────────────────────────────
#  SEQUENCE BUILDER
#  Uses a SEPARATE Close-only scaler so we can
#  inverse-transform predictions cleanly.
#  X : (samples, lookback, n_features)  — indicators only
#  y : next-day scaled Close
# ──────────────────────────────────────────────
def make_sequences(feat_scaled: np.ndarray,
                   close_values: np.ndarray,
                   lookback: int = 30):
    X, y = [], []

    for i in range(lookback, len(feat_scaled) - 1):

        current_close = close_values[i]
        next_close = close_values[i + 1]

        future_return = np.log(
            next_close / (current_close + 1e-10)
        )

        X.append(feat_scaled[i - lookback:i])
        y.append(future_return)

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
#  BUILD MODEL
# ──────────────────────────────────────────────
def build_model(n_features: int, lookback: int = 30) -> Sequential:

    model = Sequential([

        Bidirectional(
            LSTM(256, return_sequences=True),
            input_shape=(lookback, n_features)
        ),

        Dropout(0.3),

        BatchNormalization(),

        Bidirectional(
            LSTM(128, return_sequences=True)
        ),

        Dropout(0.3),

        BatchNormalization(),

        LSTM(64),

        Dropout(0.3),

        Dense(64, activation='relu'),

        Dense(32, activation='relu'),

        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='huber',
        metrics=['mae']
    )

    return model


# ──────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ──────────────────────────────────────────────
def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    data  = data.squeeze()
    delta = data.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def get_signal(rsi_value: float) -> str:
    if rsi_value < 30: return "BUY"
    if rsi_value > 70: return "SELL"
    return "HOLD"


def get_price_changes(hist_df: pd.DataFrame) -> dict:
    current_p  = float(hist_df['Close'].iloc[-1].squeeze())
    prev_day   = float(hist_df['Close'].iloc[-2].squeeze())
    prev_week  = float(hist_df['Close'].iloc[-7].squeeze()) if len(hist_df) > 7 else current_p
    return {
        "current":    current_p,
        "change_24h": ((current_p - prev_day)  / prev_day)  * 100,
        "change_7d":  ((current_p - prev_week) / prev_week) * 100,
    }


# ──────────────────────────────────────────────
#  LSTM PREDICTION  v5 — clean two-scaler design
#
#  feat_scaler : scales the 17 indicator features
#  close_scaler: scales Close price ONLY
#  → inverse_transform on close_scaler gives a
#    clean, accurate price. No mixing of scales.
# ──────────────────────────────────────────────
def run_prediction(hist_df: pd.DataFrame,
                   symbol: str = "",
                   master_df: pd.DataFrame = None) -> dict:
    LOOKBACK = 30

    fear_greed = fetch_fear_greed(n_days=3000)
    btc_dom    = fetch_btc_dominance(pd.to_datetime(hist_df.index))

    featured   = build_features(hist_df,
                                 symbol=symbol,
                                 master_df=master_df,
                                 fear_greed_series=fear_greed,
                                 btc_dom_series=btc_dom)

    feat_cols  = [c for c in featured.columns if c != 'Close']
    n_features = len(feat_cols)

    # Scaler for features only (we predict returns)
    feat_scaler  = RobustScaler()

    feat_scaled  = feat_scaler.fit_transform(featured[feat_cols].values)
    close_raw    = featured['Close'].values

    X, y = make_sequences(
        feat_scaled,
        close_raw,
        LOOKBACK
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Model file handling
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    safe_sym = symbol if symbol else 'generic'
    model_path = os.path.join(model_dir, f"{safe_sym}_model.keras")

    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
        except Exception:
            model = build_model(n_features, LOOKBACK)
    else:
        model = build_model(n_features, LOOKBACK)

    # Train with a dedicated train/val split (no random shuffle)
    split = int(len(X) * 0.9)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    sample_weights = np.linspace(
        0.5,
        1.0,
        len(X_train)
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        sample_weight=sample_weights,
        batch_size=32,
        epochs=300,
        callbacks=callbacks,
        verbose=0,
        shuffle=False
    )

    # Save model back
    try:
        model.save(model_path)
    except Exception:
        pass

    # Predict next-day return using last LOOKBACK rows of features
    last_feat   = feat_scaled[-LOOKBACK:].reshape(1, LOOKBACK, n_features)
    predicted_return = float(model.predict(last_feat, verbose=0)[0][0])

    # Convert return → price
    last_close = float(featured['Close'].iloc[-1])

    pred_price = last_close * np.exp(predicted_return)

    # Sanity: clip to reasonable range
    pred_price = float(np.clip(pred_price,
                               last_close * 0.60,
                               last_close * 1.40))
    pct_change = (pred_price / last_close - 1) * 100

    return {
        "price":        pred_price,
        "pct_change":   pct_change,
        "model":        model,
        "feat_scaler":  feat_scaler,
        "close_scaler": None,
        "featured":     featured,
        "feat_cols":    feat_cols,
    }


# ──────────────────────────────────────────────
#  FAST VALIDATION — zero retraining
# ──────────────────────────────────────────────
def run_validation(hist_df: pd.DataFrame,
                   cached_model=None,
                   cached_scaler=None,       # kept for API compat, unused
                   cached_featured=None,
                   cached_feat_scaler=None,
                   cached_close_scaler=None,
                   cached_feat_cols=None) -> dict:
    TEST_SIZE = 30
    LOOKBACK  = 30

    # Quick check: require at least 5 completed prediction rows (Actual != Pending)
    # Search user prediction history files for recent completed entries
    try:
        completed = 0
        from glob import glob
        import re
        files = glob(os.path.join(os.getcwd(), "*_prediction_history.txt"))
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        for f in files:
            try:
                with open(f, 'r') as fh:
                    for line in fh:
                        if 'Actual:' in line and 'Pending' not in line and 'pred_date:' in line:
                            m = re.search(r'pred_date:(\d{4}-\d{2}-\d{2})', line)
                            if m:
                                pd_date = pd.to_datetime(m.group(1))
                                if pd_date >= cutoff:
                                    completed += 1
                            else:
                                completed += 1
            except Exception:
                continue
        if completed < 5:
            return {
                "mape": None,
                "is_accurate": False,
                "comp_df": pd.DataFrame(),
                "message": "Not enough new validation data."
            }
    except Exception:
        # If any error while checking history, continue to validation but keep going
        pass

    featured  = cached_featured if cached_featured is not None \
                else build_features(hist_df)

    feat_cols = cached_feat_cols if cached_feat_cols is not None \
                else [c for c in featured.columns if c != 'Close']
    n_features = len(feat_cols)

    close_vals = featured['Close'].squeeze().values
    actuals    = close_vals[-TEST_SIZE:]

    if (cached_model is not None
            and cached_feat_scaler is not None
            and cached_feat_cols is not None):

        feat_scaled = cached_feat_scaler.transform(featured[feat_cols].values)
        preds = []

        for i in range(TEST_SIZE):
            end_idx    = len(feat_scaled) - TEST_SIZE + i
            seq_in     = feat_scaled[end_idx - LOOKBACK:end_idx]
            seq_in     = seq_in.reshape(1, LOOKBACK, n_features)
            pred_ret   = float(cached_model.predict(seq_in, verbose=0)[0][0])
            base       = float(close_vals[-(TEST_SIZE - i) - 1])
            pred_price = float(np.clip(base * np.exp(pred_ret), base * 0.60, base * 1.40))
            preds.append(pred_price)
    else:
        # EMA baseline fallback
        ema = featured['Close'].squeeze().ewm(span=10, adjust=False).mean().shift(1)
        preds = ema.values[-TEST_SIZE:].tolist()

    actuals = np.array(actuals, dtype=float)
    preds   = np.array(preds,   dtype=float)
    mask    = ~(np.isnan(actuals) | np.isnan(preds))
    actuals, preds = actuals[mask], preds[mask]
    mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-10))) * 100)

    return {
        "mape":        mape,
        "comp_df":     pd.DataFrame({
            'Actual':    actuals.flatten(),
            'Predicted': preds.flatten(),
            'Deviation': np.abs(actuals - preds).flatten(),
        }),
        "is_accurate": mape < 5,
    }


# ──────────────────────────────────────────────
#  TOP MOVERS
# ──────────────────────────────────────────────
def get_top_gainers(master_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return master_df.nlargest(n, '24h')[['Coin Name', 'Symbol', '24h']]


def get_top_losers(master_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return master_df.nsmallest(n, '24h')[['Coin Name', 'Symbol', '24h']]


# ══════════════════════════════════════════════
#  ANALYSIS SUITE
# ══════════════════════════════════════════════

def get_ath_atl(hist_df: pd.DataFrame) -> dict:
    close    = hist_df['Close'].squeeze()
    ath, atl = float(close.max()), float(close.min())
    current  = float(close.iloc[-1])
    return {
        "ath": ath, "atl": atl,
        "ath_date": str(close.idxmax())[:10],
        "atl_date": str(close.idxmin())[:10],
        "pct_from_ath": ((current - ath) / ath) * 100,
        "pct_from_atl": ((current - atl) / atl) * 100,
        "current": current,
    }


def get_monthly_performance(hist_df: pd.DataFrame) -> pd.DataFrame:
    df = hist_df[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    df['Close'] = df['Close'].squeeze()
    monthly     = df['Close'].resample('ME').last()
    monthly_ret = (monthly.pct_change() * 100).dropna()
    result = pd.DataFrame({
        'Year':     monthly_ret.index.year,
        'Month':    monthly_ret.index.strftime('%b'),
        'MonthNum': monthly_ret.index.month,
        'Return':   monthly_ret.values.round(2),
    })
    def label(r):
        if r >= 20:  return "🚀 Very High"
        if r >= 5:   return "📈 High"
        if r >= 0:   return "🟢 Slight Gain"
        if r >= -5:  return "🟡 Slight Loss"
        if r >= -20: return "📉 Low"
        return "🔴 Very Low"
    result['Label'] = result['Return'].apply(label)
    return result.reset_index(drop=True)


def get_volatility_analysis(hist_df: pd.DataFrame) -> pd.DataFrame:
    close       = hist_df['Close'].squeeze()
    rolling_vol = close.pct_change().rolling(30).std() * np.sqrt(365) * 100
    vol_df = pd.DataFrame({'Date': rolling_vol.index,
                           'Volatility': rolling_vol.values}).dropna()
    vol_df['Date'] = pd.to_datetime(vol_df['Date'])
    return vol_df


def get_volume_anomalies(hist_df: pd.DataFrame) -> pd.DataFrame:
    df = hist_df.copy()
    df['Vol_MA30']  = df['Volume'].squeeze().rolling(30).mean()
    df['Vol_Ratio'] = df['Volume'].squeeze() / (df['Vol_MA30'] + 1e-10)
    close_s         = df['Close'].squeeze()
    df['Direction'] = np.where(
        close_s > close_s.shift(1), "📈 Bullish Surge", "📉 Bearish Dump"
    )
    return df[df['Vol_Ratio'] > 2.0][['Close', 'Volume', 'Vol_Ratio', 'Direction']].copy().tail(30)


def get_support_resistance(hist_df: pd.DataFrame, window: int = 20) -> dict:
    close   = hist_df['Close'].squeeze()
    current = float(close.iloc[-1])
    local_max = close[(close.shift(1) < close) & (close.shift(-1) < close)]
    local_min = close[(close.shift(1) > close) & (close.shift(-1) > close)]
    resistance = sorted([float(v) for v in local_max if v > current],
                        key=lambda x: abs(x - current))[:3]
    support    = sorted([float(v) for v in local_min if v < current],
                        key=lambda x: abs(x - current))[:3]
    return {"current": current, "resistance": resistance, "support": support}


def get_trend_analysis(hist_df: pd.DataFrame) -> dict:
    close   = hist_df['Close'].squeeze()
    ema200  = close.ewm(span=200, adjust=False).mean()
    ema50   = close.ewm(span=50,  adjust=False).mean()
    ema20   = close.ewm(span=20,  adjust=False).mean()
    current = float(close.iloc[-1])
    e200, e50, e20 = float(ema200.iloc[-1]), float(ema50.iloc[-1]), float(ema20.iloc[-1])
    if current > e200 and e50 > e200:
        trend, trend_color = "🐂 Bull Market", "green"
    elif current < e200 and e50 < e200:
        trend, trend_color = "🐻 Bear Market", "red"
    else:
        trend, trend_color = "↔ Sideways / Consolidation", "orange"
    streak = 0
    for val in reversed((close > ema200).values):
        if val == (close.iloc[-1] > ema200.iloc[-1]): streak += 1
        else: break
    return {
        "trend": trend, "trend_color": trend_color,
        "trend_strength": abs((e20 - e50) / (e50 + 1e-10)) * 100,
        "ema20": e20, "ema50": e50, "ema200": e200,
        "ema_streak_days": streak, "current": current,
    }


def get_price_zones(hist_df: pd.DataFrame) -> pd.DataFrame:
    close = hist_df['Close'].squeeze()
    p10, p25 = float(close.quantile(0.10)), float(close.quantile(0.25))
    p75, p90 = float(close.quantile(0.75)), float(close.quantile(0.90))
    def zone(v):
        if v >= p90: return "ATH Zone"
        if v >= p75: return "High"
        if v >= p25: return "Normal"
        if v >= p10: return "Low"
        return "ATL Zone"
    return pd.DataFrame({
        'Date':  pd.to_datetime(hist_df.index),
        'Close': close.values,
        'Zone':  close.apply(zone).values,
    })