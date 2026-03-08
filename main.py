import os
import cohere
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from scipy.optimize import linprog

from fastapi.responses import StreamingResponse
import json

# ── App Setup ──────────────────────────────────────────
app = FastAPI(title="GridMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model & Data Once at Startup ─────────────────
with open("gridmind_model.pkl", "rb") as f:
    model = pickle.load(f)

energy = pd.read_csv("energy_dataset.csv")
energy["time"] = pd.to_datetime(energy["time"], utc=True)
energy = energy.set_index("time")

df = energy[["price actual"]].copy()
df["price actual"] = df["price actual"].ffill()

df["hour"]           = df.index.hour
df["dayofweek"]      = df.index.dayofweek
df["month"]          = df.index.month
df["is_weekend"]     = (df.index.dayofweek >= 5).astype(int)
df["price_lag_1h"]   = df["price actual"].shift(1)
df["price_lag_24h"]  = df["price actual"].shift(24)
df["price_lag_168h"] = df["price actual"].shift(168)
df["rolling_mean_24h"] = df["price actual"].rolling(24).mean()
df["rolling_std_24h"]  = df["price actual"].rolling(24).std()
df["rolling_mean_7d"]  = df["price actual"].rolling(168).mean()
df.dropna(inplace=True)

FEATURES = [
    "hour", "dayofweek", "month", "is_weekend",
    "price_lag_1h", "price_lag_24h", "price_lag_168h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_7d"
]

df["predicted_price"] = model.predict(df[FEATURES])
df["next_predicted"]  = df["predicted_price"].shift(-1)
df["price_change_pct"] = (
    (df["next_predicted"] - df["predicted_price"]) / df["predicted_price"] * 100
)

def get_signal(change_pct):
    if change_pct > 2:   return "BUY"
    elif change_pct < -2: return "SELL"
    else:                 return "HOLD"

df["signal"] = df["price_change_pct"].apply(get_signal)
df.dropna(inplace=True)

print("✅ Model & data loaded successfully!")

# ── Request Models ─────────────────────────────────────
class OptimizeRequest(BaseModel):
    total_mwh: float = 100.0
    max_per_hour: float = 10.0

# ── Routes ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "GridMind API is running 🚀", "version": "1.0.0"}


@app.get("/forecast")
def get_forecast():
    """Return last 7 days of actual vs predicted prices"""
    week = df.tail(168).copy()
    week = week.reset_index()
    return {
        "labels": week["time"].astype(str).tolist(),
        "actual": week["price actual"].round(2).tolist(),
        "predicted": week["predicted_price"].round(2).tolist(),
        "model_r2": 0.9619,
        "model_mae": 1.68
    }


@app.get("/signals")
def get_signals():
    """Return last 7 days of trading signals"""
    week = df.tail(168).copy()
    week = week.reset_index()
    
    signal_counts = df["signal"].value_counts()
    
    return {
        "labels"   : week["time"].astype(str).tolist(),
        "actual"   : week["price actual"].round(2).tolist(),
        "predicted": week["predicted_price"].round(2).tolist(),
        "signals"  : week["signal"].tolist(),
        "summary"  : {
            "BUY" : int(signal_counts.get("BUY",  0)),
            "SELL": int(signal_counts.get("SELL", 0)),
            "HOLD": int(signal_counts.get("HOLD", 0)),
        }
    }


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """Return optimal energy buying schedule"""
    week = df.tail(168).copy()
    predicted_prices = week["predicted_price"].values
    n_hours = len(predicted_prices)

    c      = predicted_prices
    A_eq   = np.ones((1, n_hours))
    b_eq   = np.array([req.total_mwh])
    bounds = [(0, req.max_per_hour)] * n_hours

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        schedule     = result.x
        total_cost   = float(result.fun)
        naive_cost   = float(np.dot(
            np.full(n_hours, req.total_mwh / n_hours), predicted_prices
        ))
        savings      = naive_cost - total_cost
        savings_pct  = (savings / naive_cost) * 100

        week = week.reset_index()
        return {
            "total_cost"     : round(total_cost, 2),
            "naive_cost"     : round(naive_cost, 2),
            "savings"        : round(savings, 2),
            "savings_pct"    : round(savings_pct, 2),
            "avg_price_paid" : round(total_cost / req.total_mwh, 2),
            "market_avg"     : round(float(predicted_prices.mean()), 2),
            "schedule": [
                {
                    "datetime" : str(week["time"].iloc[i]),
                    "price"    : round(float(predicted_prices[i]), 2),
                    "buy_mwh"  : round(float(schedule[i]), 2)
                }
                for i in range(n_hours)
            ]
        }
    else:
        return {"error": "Optimization failed"}
    
class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    co = cohere.ClientV2(os.environ.get("COHERE_API_KEY"))
    
    messages = [
        {
            "role": "system",
            "content": """You are GridMind AI, an expert European energy markets analyst built by Rahul, a Data Science student. 
You were created by Rahul as a portfolio project to demonstrate European energy price forecasting, trading signals, and portfolio optimization.
If anyone asks who created you, say: 'I was built by Rahul, a Data Science student specializing in AI and energy analytics.'
Be concise, insightful, and professional in all energy market discussions."""
        }
    ] + req.history + [
        {"role": "user", "content": req.message}
    ]
    
    def stream_response():
        for event in co.chat_stream(
            model="command-r-plus-08-2024",
            messages=messages
        ):
            if event.type == "content-delta":
                token = event.delta.message.content.text
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/stats")
def get_stats():
    """Return overall dataset statistics"""
    return {
        "total_hours"  : len(df),
        "date_from"    : str(df.index.min()),
        "date_to"      : str(df.index.max()),
        "avg_price"    : round(float(df["price actual"].mean()), 2),
        "min_price"    : round(float(df["price actual"].min()), 2),
        "max_price"    : round(float(df["price actual"].max()), 2),
        "model_r2"     : 0.9619,
        "model_mae"    : 1.68,
        "signal_counts": df["signal"].value_counts().to_dict()
    }