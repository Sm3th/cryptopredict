"""
CryptoPredict - Model Training Script
Trains an LSTM model on historical cryptocurrency price data.

Usage:
    python train_model.py --coin bitcoin --days 730
    python train_model.py --coin ethereum --days 365 --seq_length 30
"""

import argparse
import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required (Docker/CI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore")

# ============================================================================
# DATA COLLECTION
# ============================================================================

def fetch_crypto_data(coin_id: str = "bitcoin", days: int = 730) -> pd.DataFrame:
    """Fetch OHLCV data from CoinGecko (free tier) with retry on 429."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    print(f"→ Fetching {coin_id} data ({days} days)...")
    for attempt in range(5):
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  ⚠ Rate limited. Waiting {wait}s before retry ({attempt + 1}/5)...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    data = response.json()

    df = pd.DataFrame({
        "timestamp": [p[0] for p in data["prices"]],
        "price":     [p[1] for p in data["prices"]],
        "volume":    [v[1] for v in data["total_volumes"]],
        "market_cap":[m[1] for m in data["market_caps"]],
    })
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").drop("timestamp", axis=1)
    print(f"  ✓ {len(df)} rows fetched  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical analysis indicators as model features."""
    df = df.copy()

    # Moving averages
    for w in [7, 21, 50]:
        df[f"MA_{w}"] = df["price"].rolling(w).mean()

    # RSI
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # Bollinger Bands
    df["BB_mid"]   = df["price"].rolling(20).mean()
    df["BB_upper"] = df["BB_mid"] + 2 * df["price"].rolling(20).std()
    df["BB_lower"] = df["BB_mid"] - 2 * df["price"].rolling(20).std()

    # Volatility & momentum
    df["volatility"]    = df["price"].rolling(7).std()
    df["price_change"]  = df["price"].pct_change()
    df["volume_change"] = df["volume"].pct_change()
    df["momentum"]      = df["price"] - df["price"].shift(10)

    df = df.dropna()
    print(f"  ✓ Technical indicators added. Final shape: {df.shape}")
    return df


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_sequences(data: np.ndarray, seq_length: int = 60):
    """Build sliding-window sequences for LSTM input."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ============================================================================
# MODEL
# ============================================================================

def build_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    print("  ✓ LSTM model built")
    return model


# ============================================================================
# FUTURE PREDICTION
# ============================================================================

def predict_future(model, scaler: MinMaxScaler, last_sequence: np.ndarray, days: int = 7) -> np.ndarray:
    """Autoregressive multi-step forecast via sliding window."""
    predictions = []
    seq = last_sequence.copy()
    for _ in range(days):
        pred = model.predict(seq.reshape(1, -1, 1), verbose=0)[0, 0]
        predictions.append(pred)
        seq = np.append(seq[1:], pred)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train(coin_id: str = "bitcoin", days: int = 730, seq_length: int = 60, epochs: int = 50, models_dir: str = "models"):
    print("\n" + "=" * 60)
    print("  CRYPTOPREDICT — MODEL TRAINING")
    print("=" * 60)

    os.makedirs(models_dir, exist_ok=True)

    # 1. Data
    df = fetch_crypto_data(coin_id, days)
    df = add_technical_indicators(df)

    # 2. Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    price_data = df["price"].values.reshape(-1, 1)
    scaled = scaler.fit_transform(price_data)

    # 3. Split
    train_size = int(len(scaled) * 0.8)
    X_train, y_train = create_sequences(scaled[:train_size], seq_length)
    X_test,  y_test  = create_sequences(scaled[train_size:],  seq_length)
    print(f"\n  Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. Build & train
    model = build_lstm_model((X_train.shape[1], 1))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(models_dir, f"{coin_id}_best.keras"), monitor="val_loss", save_best_only=True, verbose=0),
    ]

    print(f"\n  Training for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Evaluate
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # 6. Visualise
    _plot_results(df, actuals, preds, history, coin_id)

    # 7. Save
    model_path  = os.path.join(models_dir, f"{coin_id}_model.keras")
    scaler_path = os.path.join(models_dir, f"{coin_id}_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n  ✓ Model  → {model_path}")
    print(f"  ✓ Scaler → {scaler_path}")

    return model, scaler, df


def _plot_results(df, actuals, preds, history, coin_id: str):
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{coin_id.upper()} — Training Report", fontsize=16, fontweight="bold")

    # Price history
    ax = axes[0, 0]
    ax.plot(df.index, df["price"], label="Close", linewidth=1.5)
    ax.plot(df.index, df["MA_7"],  label="MA 7",  alpha=0.7, linewidth=1)
    ax.plot(df.index, df["MA_21"], label="MA 21", alpha=0.7, linewidth=1)
    ax.set_title("Price History")
    ax.set_ylabel("USD")
    ax.legend()

    # Actual vs Predicted
    ax = axes[0, 1]
    ax.plot(actuals, label="Actual",    linewidth=1.5)
    ax.plot(preds,   label="Predicted", linewidth=1.5, alpha=0.8)
    ax.set_title("Test Set: Actual vs Predicted")
    ax.set_ylabel("USD")
    ax.legend()

    # Training loss
    ax = axes[1, 0]
    ax.plot(history.history["loss"],     label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    # RSI
    ax = axes[1, 1]
    ax.plot(df.index, df["RSI"], color="orange", linewidth=1.2)
    ax.axhline(70, color="red",   linestyle="--", alpha=0.6, label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", alpha=0.6, label="Oversold (30)")
    ax.set_title("RSI (14)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{coin_id}_training_report.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Plot saved → {coin_id}_training_report.png")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CryptoPredict LSTM model")
    parser.add_argument("--coin",       type=str, default="bitcoin", help="CoinGecko coin ID")
    parser.add_argument("--days",       type=int, default=730,       help="Days of historical data")
    parser.add_argument("--seq_length", type=int, default=60,        help="Lookback window (days)")
    parser.add_argument("--epochs",     type=int, default=50,        help="Max training epochs")
    parser.add_argument("--models_dir", type=str, default=os.getenv("MODELS_DIR", "models"), help="Directory to save models")
    args = parser.parse_args()

    model, scaler, df = train(
        coin_id=args.coin,
        days=args.days,
        seq_length=args.seq_length,
        epochs=args.epochs,
        models_dir=args.models_dir,
    )

    # 7-day forecast
    last_seq = scaler.transform(df["price"].values[-args.seq_length:].reshape(-1, 1)).flatten()
    future   = predict_future(model, scaler, last_seq, days=7)
    current  = df["price"].iloc[-1]

    print("\n" + "=" * 60)
    print("  NEXT 7-DAY FORECAST")
    print("=" * 60)
    print(f"  Current: ${current:,.2f}")
    for i, p in enumerate(future, 1):
        chg = ((p - current) / current) * 100
        print(f"  Day {i}:  ${p:,.2f}  ({chg:+.2f}%)")

    print("\n  ✓ Done. Next: start the FastAPI backend with  uvicorn main:app --reload")
