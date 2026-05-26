"""
CryptoPredict - FastAPI Backend
AI-powered cryptocurrency prediction platform with active learning feedback loop.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
import requests
import joblib
import json
import os
from collections import deque

# ============================================================================
# MODELS & SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    coin_id: str = "bitcoin"
    days: int = 7

class PredictionResponse(BaseModel):
    coin_id: str
    current_price: float
    predictions: List[Dict[str, Any]]
    confidence: float
    sentiment: str
    sentiment_score: float
    model_accuracy: float

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_price: float
    predicted_price: float
    date: str
    is_accurate: bool
    user_comment: Optional[str] = None

class RetrainRequest(BaseModel):
    coin_id: str = "bitcoin"
    days: int = 730

class MetricsResponse(BaseModel):
    total_predictions: int
    accurate_predictions: int
    accuracy_percentage: float
    mae: float
    rmse: float
    last_retrain: str
    feedback_count: int

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="CryptoPredict API",
    description="AI-powered cryptocurrency prediction with active learning",
    version="1.0.0"
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class ModelState:
    """Thread-safe global model state management."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feedback_queue: deque = deque(maxlen=100)
        self._lock = asyncio.Lock()
        self.metrics: Dict[str, Any] = {
            "total_predictions": 0,
            "accurate_predictions": 0,
            "mae_sum": 0.0,
            "rmse_sum": 0.0,
            "last_retrain": "Never",
            "feedback_count": 0,
        }
        self.retrain_in_progress = False
        self._load_model()
        self._load_feedback()
        self._load_metrics()

    def _load_model(self):
        try:
            model_path = os.getenv("MODEL_PATH", "crypto_lstm_model.keras")
            scaler_path = os.getenv("SCALER_PATH", "scaler.pkl")
            # Also support legacy .h5
            if not os.path.exists(model_path):
                model_path = "crypto_lstm_model.h5"
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                from tensorflow import keras
                self.model = keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                print("✓ Model loaded successfully")
            else:
                print("⚠ No pre-trained model found. Train via train_model.py first.")
        except Exception as e:
            print(f"✗ Error loading model: {e}")

    def _safe_load_json(self, path: str, default):
        """Load JSON from file, returning default on any error."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠ Could not load {path}: {e}")
        return default

    def _load_feedback(self):
        data = self._safe_load_json("feedback.json", [])
        self.feedback_queue = deque(data, maxlen=100)

    def _load_metrics(self):
        data = self._safe_load_json("metrics.json", None)
        if data:
            self.metrics.update(data)

    def save_feedback(self):
        with open("feedback.json", "w") as f:
            json.dump(list(self.feedback_queue), f, indent=2)

    def save_metrics(self):
        with open("metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)


state = ModelState()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_crypto_data(coin_id: str = "bitcoin", days: int = 365) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        prices = data["prices"]
        volumes = data.get("total_volumes", [[p[0], 0] for p in prices])

        df = pd.DataFrame({
            "timestamp": [p[0] for p in prices],
            "price": [p[1] for p in prices],
            "volume": [v[1] for v in volumes],
        })
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("date").drop("timestamp", axis=1)
        return df
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="CoinGecko API timeout. Try again.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"CoinGecko API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch error: {str(e)}")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA_7"] = df["price"].rolling(window=7).mean()
    df["MA_21"] = df["price"].rolling(window=21).mean()

    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["volatility"] = df["price"].rolling(window=7).std()
    df["price_change"] = df["price"].pct_change()
    return df.dropna()


def create_sequences(data: np.ndarray, seq_length: int = 60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def predict_future(model, scaler, last_sequence: np.ndarray, days: int = 7) -> np.ndarray:
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        current_sequence = np.append(current_sequence[1:], pred[0, 0])
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def get_sentiment_score(coin_id: str = "bitcoin"):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        response = requests.get(url, timeout=10)
        data = response.json()
        score = data.get("sentiment_votes_up_percentage", 50) or 50
        if score >= 60:
            return "Bullish", round(score, 1)
        elif score >= 40:
            return "Neutral", round(score, 1)
        else:
            return "Bearish", round(score, 1)
    except Exception:
        return "Unknown", 50.0


def calculate_confidence(predictions: np.ndarray, historical_prices: np.ndarray) -> float:
    volatility = np.std(historical_prices[-30:]) / np.mean(historical_prices[-30:])
    if volatility < 0.05:
        return 0.85
    elif volatility < 0.10:
        return 0.70
    elif volatility < 0.15:
        return 0.55
    return 0.40

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "version": "1.0.0",
        "model_loaded": state.model is not None,
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": state.model is not None,
        "scaler_loaded": state.scaler is not None,
        "feedback_count": len(state.feedback_queue),
        "total_predictions": state.metrics["total_predictions"],
        "retrain_in_progress": state.retrain_in_progress,
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if state.model is None or state.scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py first to generate the model.",
        )
    try:
        df = fetch_crypto_data(request.coin_id, days=365)
        df = add_technical_indicators(df)
        current_price = float(df["price"].iloc[-1])

        price_data = df["price"].values
        scaled_data = state.scaler.transform(price_data.reshape(-1, 1)).flatten()
        last_60 = scaled_data[-60:]

        raw_predictions = predict_future(state.model, state.scaler, last_60, days=request.days)
        sentiment, sentiment_score = get_sentiment_score(request.coin_id)
        confidence = calculate_confidence(raw_predictions, price_data)

        prediction_list = []
        for i, pred in enumerate(raw_predictions, 1):
            date = datetime.now() + timedelta(days=i)
            change = ((pred - current_price) / current_price) * 100
            prediction_list.append({
                "day": i,
                "date": date.strftime("%Y-%m-%d"),
                "price": round(float(pred), 2),
                "change_percentage": round(float(change), 2),
            })

        async with state._lock:
            state.metrics["total_predictions"] += 1
        state.save_metrics()

        feedback_count = state.metrics["feedback_count"]
        accurate = state.metrics["accurate_predictions"]
        model_accuracy = round((accurate / feedback_count) * 100, 2) if feedback_count > 0 else 0.0

        return PredictionResponse(
            coin_id=request.coin_id,
            current_price=round(current_price, 2),
            predictions=prediction_list,
            confidence=confidence,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            model_accuracy=model_accuracy,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        error = abs(feedback.actual_price - feedback.predicted_price)
        feedback_data = {
            "prediction_id": feedback.prediction_id,
            "actual_price": feedback.actual_price,
            "predicted_price": feedback.predicted_price,
            "date": feedback.date,
            "is_accurate": feedback.is_accurate,
            "user_comment": feedback.user_comment,
            "timestamp": datetime.now().isoformat(),
            "error": round(error, 2),
        }

        async with state._lock:
            state.feedback_queue.append(feedback_data)
            state.metrics["feedback_count"] += 1
            if feedback.is_accurate:
                state.metrics["accurate_predictions"] += 1
            state.metrics["mae_sum"] += error
            state.metrics["rmse_sum"] += error ** 2

        state.save_feedback()
        state.save_metrics()

        fc = state.metrics["feedback_count"]
        accuracy = round((state.metrics["accurate_predictions"] / fc) * 100, 2) if fc > 0 else 0.0

        return {
            "status": "success",
            "message": "Feedback recorded. Thank you!",
            "feedback_count": fc,
            "should_retrain": fc >= 50 and fc % 50 == 0,
            "current_accuracy": accuracy,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@app.post("/api/retrain")
async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    if state.retrain_in_progress:
        raise HTTPException(status_code=409, detail="Retraining already in progress.")
    if state.scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded. Train initial model first.")

    def retrain_task():
        try:
            state.retrain_in_progress = True
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping

            df = fetch_crypto_data(request.coin_id, days=request.days)
            df = add_technical_indicators(df)

            price_data = df["price"].values.reshape(-1, 1)
            scaled_data = state.scaler.transform(price_data)

            X, y = create_sequences(scaled_data, seq_length=60)
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]

            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0,
            )

            model.save("crypto_lstm_model.keras")
            state.model = model
            state.metrics["last_retrain"] = datetime.now().isoformat()
            state.save_metrics()
            print("✓ Model retrained successfully")
        except Exception as e:
            print(f"✗ Retrain failed: {e}")
        finally:
            state.retrain_in_progress = False

    background_tasks.add_task(retrain_task)
    return {
        "status": "started",
        "message": "Model retraining started in background.",
        "estimated_time": "2–5 minutes",
        "feedback_used": len(state.feedback_queue),
    }


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    fc = state.metrics["feedback_count"]
    accuracy = round((state.metrics["accurate_predictions"] / fc) * 100, 2) if fc > 0 else 0.0
    mae = round(state.metrics["mae_sum"] / fc, 2) if fc > 0 else 0.0
    rmse = round(np.sqrt(state.metrics["rmse_sum"] / fc), 2) if fc > 0 else 0.0

    return MetricsResponse(
        total_predictions=state.metrics["total_predictions"],
        accurate_predictions=state.metrics["accurate_predictions"],
        accuracy_percentage=accuracy,
        mae=mae,
        rmse=rmse,
        last_retrain=state.metrics["last_retrain"],
        feedback_count=fc,
    )


@app.get("/api/feedback/history")
async def get_feedback_history():
    return {
        "total_feedback": len(state.feedback_queue),
        "feedback": list(state.feedback_queue),
    }


@app.get("/api/coins")
async def get_supported_coins():
    return {
        "supported_coins": [
            {"id": "bitcoin",  "name": "Bitcoin",  "symbol": "BTC"},
            {"id": "ethereum", "name": "Ethereum", "symbol": "ETH"},
            {"id": "cardano",  "name": "Cardano",  "symbol": "ADA"},
            {"id": "solana",   "name": "Solana",   "symbol": "SOL"},
            {"id": "ripple",   "name": "Ripple",   "symbol": "XRP"},
        ]
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  CRYPTOPREDICT API  —  http://localhost:8000")
    print("  Swagger UI         —  http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
