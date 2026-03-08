# ⚡ GridMind — European Energy Trading Intelligence Platform

![GridMind Banner](https://img.shields.io/badge/GridMind-Energy%20Intelligence-4a9eff?style=for-the-badge&logo=lightning&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-R²%200.9619-orange?style=flat-square)
![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)
![Deployed](https://img.shields.io/badge/Status-Live-00ff9d?style=flat-square)

> An end-to-end AI-powered platform for European electricity price forecasting, algorithmic trading signals, and portfolio optimization — built to demonstrate real-world data science applications in the energy sector.

🌐 **Live Demo:** [gridmind-frontend-npgc.vercel.app](https://gridmind-frontend-npgc.vercel.app)  
⚙️ **API Docs:** [gridmind-backend.onrender.com/docs](https://gridmind-backend.onrender.com/docs)

---

## 📸 Preview

| Dashboard | Trading Signals | Portfolio Optimizer | AI Chat |
|-----------|----------------|--------------------|---------| 
| Real-time stats | Buy/Sell/Hold signals | Linear programming | Streaming AI |

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Model R² Score** | 0.9619 |
| **MAE (Mean Absolute Error)** | €1.68/MWh |
| **Portfolio Savings** | 18.9% vs naive buying |
| **Dataset Size** | 35,000+ hours of market data |
| **Trading Signals** | 34,895 BUY/SELL/HOLD signals generated |

---

## 🧠 Features

### 📈 Price Forecasting
- XGBoost regression model trained on 4 years of Spanish electricity market data
- Features include lag variables (1h, 24h, 168h), rolling statistics, and time-based features
- R² = 0.9619 — predicts prices within €1.68/MWh on average

### ⚡ Trading Signals
- Algorithmic BUY/SELL/HOLD signal generation based on predicted price movements
- BUY when predicted price change > +2%, SELL when < -2%, HOLD otherwise
- 34,895 signals generated across the full dataset

### ⚙️ Portfolio Optimizer
- Linear Programming (scipy `linprog` with HiGHS solver)
- Minimizes total energy procurement cost given constraints
- Achieves **18.9% cost savings** vs naive (uniform) buying strategy
- Configurable: total MWh needed, max MWh per hour

### 🤖 AI Chat (GridMind AI)
- Powered by Cohere `command-r-plus` model
- Server-Sent Events (SSE) streaming for real-time word-by-word responses
- Contextual awareness of the platform's data and results
- Built by Rahul — answers questions about energy markets, forecasts, and trading

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Next.js Frontend                   │
│          (Vercel) gridmind-frontend-npgc.vercel.app  │
│                                                      │
│  📈 Forecast  ⚡ Signals  ⚙️ Optimize  🤖 AI Chat   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Backend                     │
│         (Render) gridmind-backend.onrender.com       │
│                                                      │
│  GET /forecast  GET /signals  POST /optimize         │
│  GET /stats     POST /chat (streaming SSE)           │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   XGBoost         Pandas          Cohere
   Model (.pkl)    DataFrame       API (LLM)
   (trained on     (35K rows       (streaming
   energy data)    energy data)     chat)
```

---

## 🛠️ Tech Stack

### Machine Learning
- **XGBoost** — Gradient boosting for price prediction
- **Scikit-learn** — Feature engineering, preprocessing
- **Pandas / NumPy** — Data manipulation
- **SciPy** — Linear programming optimization

### Backend
- **FastAPI** — High-performance Python API framework
- **Uvicorn** — ASGI server
- **Cohere API** — LLM for AI chat with streaming
- **python-dotenv** — Environment variable management

### Frontend
- **Next.js 15** — React framework
- **Recharts** — Interactive charts (AreaChart, LineChart, BarChart)
- **Tailwind CSS** — Styling
- **Server-Sent Events** — Real-time AI streaming

### Deployment
- **Vercel** — Frontend hosting (CI/CD via GitHub)
- **Render** — Backend hosting (auto-deploy on push)

---

## 📊 Dataset

- **Source:** Kaggle — Spanish Electricity Market (ESIOS)
- **Period:** 2015–2018 (4 years, hourly resolution)
- **Size:** 35,064 data points
- **Features:** electricity price (€/MWh), generation by source, weather data

---

## 🚀 Local Setup

### Backend
```bash
git clone https://github.com/Rahul-2806/gridmind-ml.git
cd gridmind-ml

pip install -r requirements.txt

# Create .env file
echo "COHERE_API_KEY=your_key_here" > .env

# Train model (optional — model.pkl included)
python train_model.py

# Start API server
python -m uvicorn main:app --reload
# API running at http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/docs
```

### Frontend
```bash
git clone https://github.com/Rahul-2806/gridmind-frontend.git
cd gridmind-frontend

npm install

# Update API URL in app/page.tsx if running locally
# const API = "http://127.0.0.1:8000"

npm run dev
# Running at http://localhost:3000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/stats` | Dataset & model statistics |
| `GET` | `/forecast` | Price predictions vs actuals |
| `GET` | `/signals` | Trading signals (BUY/SELL/HOLD) |
| `POST` | `/optimize` | Run portfolio optimizer |
| `POST` | `/chat` | Streaming AI chat (SSE) |

### Example: Portfolio Optimization
```bash
curl -X POST https://gridmind-backend.onrender.com/optimize \
  -H "Content-Type: application/json" \
  -d '{"total_mwh": 100, "max_per_hour": 10}'
```

```json
{
  "total_cost": 5472.80,
  "naive_cost": 6745.66,
  "savings": 1272.86,
  "savings_pct": "18.9",
  "avg_price_paid": 54.73
}
```

---

## 🧮 ML Model Details

### Feature Engineering
```python
features = [
    'hour', 'dayofweek', 'month', 'is_weekend',
    'price_lag_1h',      # Price 1 hour ago
    'price_lag_24h',     # Price 24 hours ago  
    'price_lag_168h',    # Price 1 week ago
    'rolling_mean_24h',  # 24-hour rolling average
    'rolling_std_24h',   # 24-hour rolling std deviation
    'rolling_mean_7d'    # 7-day rolling average
]
```

### XGBoost Hyperparameters
```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)
```

### Results
```
MAE:  €1.68/MWh
RMSE: €2.24/MWh
R²:   0.9619
```

---

## 👨‍💻 About the Developer

Built by **Rahul** — Data Science student at Datamites (IABAC & NASSCOM certified), with expertise in Python, Machine Learning, and full-stack AI application development.

- 🎯 Specialization: Energy Analytics, ML, GenAI
- 🛠️ Stack: Python, FastAPI, Next.js, XGBoost, LLMs
- 📍 Based in Kerala, India

---

## 📄 License

MIT License — feel free to use this project as a reference or template.

---

*Built with ⚡ for the European energy market*