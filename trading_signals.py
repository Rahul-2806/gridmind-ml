import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load model & data ──────────────────────────────────
with open("gridmind_model.pkl", "rb") as f:
    model = pickle.load(f)

energy = pd.read_csv("energy_dataset.csv")
energy["time"] = pd.to_datetime(energy["time"], utc=True)
energy = energy.set_index("time")

df = energy[["price actual"]].copy()
df["price actual"] = df["price actual"].ffill()

# ── Feature Engineering (same as training) ────────────
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

# ── Generate Predictions ───────────────────────────────
df["predicted_price"] = model.predict(df[FEATURES])

# ── Trading Signal Logic ───────────────────────────────
# Compare predicted next price vs current price
df["next_predicted"] = df["predicted_price"].shift(-1)
df["price_change_pct"] = (
    (df["next_predicted"] - df["predicted_price"]) / df["predicted_price"] * 100
)

def generate_signal(change_pct):
    if change_pct > 2:       # Price rising >2% → BUY (buy cheap, sell when high)
        return "BUY"
    elif change_pct < -2:    # Price falling >2% → SELL (sell before price drops)
        return "SELL"
    else:
        return "HOLD"        # Stable → HOLD

df["signal"] = df["price_change_pct"].apply(generate_signal)
df.dropna(inplace=True)

# ── Signal Summary ─────────────────────────────────────
print("="*45)
print("  GridMind — Trading Signal Summary")
print("="*45)
signal_counts = df["signal"].value_counts()
print(f"  BUY  signals : {signal_counts.get('BUY',  0)}")
print(f"  SELL signals : {signal_counts.get('SELL', 0)}")
print(f"  HOLD signals : {signal_counts.get('HOLD', 0)}")
print("="*45)

# ── Backtest: How profitable are the signals? ──────────
df["return"] = 0.0
df.loc[df["signal"] == "BUY",  "return"] =  df["price_change_pct"]
df.loc[df["signal"] == "SELL", "return"] = -df["price_change_pct"]

total_return = df["return"].sum()
avg_return   = df["return"].mean()
win_rate     = (df[df["signal"] != "HOLD"]["return"] > 0).mean() * 100

print(f"\n  Backtest Results")
print("="*45)
print(f"  Total return  : {total_return:.2f}%")
print(f"  Avg per trade : {avg_return:.4f}%")
print(f"  Win rate      : {win_rate:.1f}%")
print("="*45)

# ── Plot last 7 days with signals ──────────────────────
last_week = df.tail(168).copy()

color_map = {"BUY": "green", "SELL": "red", "HOLD": "gray"}
colors = last_week["signal"].map(color_map)

plt.figure(figsize=(16, 6))
plt.plot(last_week["price actual"].values, color="blue", alpha=0.6, label="Actual Price")
plt.plot(last_week["predicted_price"].values, color="orange", alpha=0.6, label="Predicted Price", linestyle="--")
plt.scatter(range(len(last_week)), last_week["price actual"].values,
            c=colors, s=40, zorder=5)

buy_patch  = mpatches.Patch(color="green", label="BUY Signal")
sell_patch = mpatches.Patch(color="red",   label="SELL Signal")
hold_patch = mpatches.Patch(color="gray",  label="HOLD Signal")
plt.legend(handles=[buy_patch, sell_patch, hold_patch,
           plt.Line2D([0],[0], color="blue",   label="Actual"),
           plt.Line2D([0],[0], color="orange", label="Predicted", linestyle="--")])

plt.title("GridMind — Trading Signals (Last 7 Days)")
plt.ylabel("Price (€/MWh)")
plt.xlabel("Hour")
plt.tight_layout()
plt.savefig("trading_signals.png")
plt.show()

# ── Save results ───────────────────────────────────────
df[["price actual", "predicted_price", "signal", "price_change_pct"]].to_csv("trading_results.csv")
print("\n✅ Trading signals saved to trading_results.csv")
print("🚀 Phase 3 Complete!")