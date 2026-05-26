# CryptoPredict 🔮

> AI-powered cryptocurrency price forecasting with an active learning feedback loop.

[![CI](https://github.com/Sm3th/cryptopredict/actions/workflows/ci.yml/badge.svg)](https://github.com/Sm3th/cryptopredict/actions)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **LSTM Model** | Multi-layer LSTM trained on historical price data with RSI, Bollinger Bands, and MA indicators |
| 📈 **7/14-day Forecast** | Autoregressive multi-step price prediction with confidence scoring |
| 💬 **Active Learning** | User feedback loop that tracks model accuracy and triggers retraining |
| 📊 **Live Metrics** | MAE, RMSE, and accuracy dashboard updated in real time |
| 🔄 **Auto-Retrain** | Background retraining triggered after every 50 feedback entries |
| 🌐 **CoinGecko API** | Free-tier market data — no API key required |
| 🐳 **Docker** | Single-command full-stack deployment |

---

## 📸 Screenshots

> Add screenshots here after running the app locally.

---

## 🏗 Architecture

```
cryptopredict/
├── backend/
│   ├── main.py              # FastAPI application & REST endpoints
│   ├── train_model.py       # LSTM training pipeline (run once before API)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       └── test_api.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React application
│   │   ├── utils/api.js     # Typed API client
│   │   ├── hooks/           # Custom React hooks
│   │   └── index.css        # Global styles
│   ├── Dockerfile
│   ├── nginx.conf
│   └── vite.config.js
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- (Optional) Docker & Docker Compose

---

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/Sm3th/cryptopredict.git
cd cryptopredict

# 1. Train the model first (runs outside Docker to save the .keras file)
cd backend
pip install -r requirements.txt
python train_model.py --coin bitcoin --days 730

# 2. Spin up everything
cd ..
docker-compose up --build
```

Open **http://localhost** for the app, **http://localhost:8000/docs** for Swagger UI.

---

### Option 2 — Local development

#### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model (required before starting the API)
python train_model.py --coin bitcoin --days 730

# Start API
uvicorn main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 🧠 Model Details

### Architecture

```
Input (60 × 1)
   ↓
LSTM(128) → Dropout(0.2)
   ↓
LSTM(64)  → Dropout(0.2)
   ↓
LSTM(32)  → Dropout(0.2)
   ↓
Dense(16, relu)
   ↓
Dense(1)  ← price prediction
```

### Features used

| Feature | Description |
|---|---|
| `price` | Daily closing price (normalized 0–1) |
| `MA_7 / MA_21 / MA_50` | Moving averages |
| `RSI` | Relative Strength Index (14-period) |
| `BB_upper / BB_lower` | Bollinger Bands (20-period, ±2σ) |
| `volatility` | 7-day rolling std |
| `momentum` | price − price[−10] |

### Training configuration

| Parameter | Value |
|---|---|
| Lookback window | 60 days |
| Train / Test split | 80 / 20 |
| Epochs | 50 (early stopping, patience=10) |
| Batch size | 32 |
| Optimiser | Adam |
| Loss | MSE |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/api/health` | Detailed system status |
| `POST` | `/api/predict` | Run price forecast |
| `POST` | `/api/feedback` | Submit prediction feedback |
| `POST` | `/api/retrain` | Trigger background retraining |
| `GET` | `/api/metrics` | Model performance metrics |
| `GET` | `/api/feedback/history` | All feedback records |
| `GET` | `/api/coins` | Supported cryptocurrencies |

Full interactive documentation: **http://localhost:8000/docs**

### Example — Predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"coin_id": "bitcoin", "days": 7}'
```

```json
{
  "coin_id": "bitcoin",
  "current_price": 67450.00,
  "predictions": [
    { "day": 1, "date": "2026-05-28", "price": 68120.50, "change_percentage": 0.99 },
    ...
  ],
  "confidence": 0.70,
  "sentiment": "Bullish",
  "sentiment_score": 72.4,
  "model_accuracy": 68.0
}
```

### Example — Feedback

```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "bitcoin_1234567890",
    "actual_price": 68000.00,
    "predicted_price": 68120.50,
    "date": "2026-05-28",
    "is_accurate": true
  }'
```

---

## 🧪 Tests

```bash
cd backend
pip install pytest httpx anyio
pytest tests/ -v
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:5173,...` | CORS allowed origins |
| `MODEL_PATH` | `crypto_lstm_model.keras` | Path to trained model |
| `SCALER_PATH` | `scaler.pkl` | Path to fitted scaler |
| `VITE_API_URL` | `/api` | API base URL (frontend) |

---

## 🗺 Roadmap

- [ ] Multi-coin comparison view
- [ ] Sentiment analysis from Twitter/Reddit
- [ ] Email alerts for large predicted swings
- [ ] Model versioning & rollback
- [ ] PostgreSQL for persistent feedback storage

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**. Cryptocurrency markets are highly volatile. Do **not** make financial decisions based on these predictions.

---

## 📄 License

MIT © 2026 — [İsmet Organ](https://github.com/Sm3th)
