import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ── Load model & data ──────────────────────────────────
with open("gridmind_model.pkl", "rb") as f:
    model = pickle.load(f)

energy = pd.read_csv("energy_dataset.csv")
energy["time"] = pd.to_datetime(energy["time"], utc=True)
energy = energy.set_index("time")

df = energy[["price actual"]].copy()
df["price actual"] = df["price actual"].ffill()

# ── Feature Engineering ────────────────────────────────
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

# ── Take last 7 days for optimization ─────────────────
week = df.tail(168).copy()
week = week.reset_index()
predicted_prices = week["predicted_price"].values
n_hours = len(predicted_prices)  # 168 hours

print("="*50)
print("  GridMind — Energy Portfolio Optimizer")
print("="*50)

# ── Optimization Problem ───────────────────────────────
# Goal: Buy exactly 100 MWh total over the week
# Constraint: Buy between 0 and 10 MWh per hour
# Objective: Minimize total cost
TOTAL_MWH   = 100   # total energy to buy (MWh)
MAX_PER_HOUR = 10   # max purchase per hour (MWh)
MIN_PER_HOUR = 0    # min purchase per hour (MWh)

# linprog minimizes: c @ x
# c = predicted prices (cost per MWh each hour)
c = predicted_prices

# Equality constraint: sum of all purchases = TOTAL_MWH
A_eq = np.ones((1, n_hours))
b_eq = np.array([TOTAL_MWH])

# Bounds: each hour between 0 and MAX_PER_HOUR
bounds = [(MIN_PER_HOUR, MAX_PER_HOUR)] * n_hours

print(f"\n  Problem Setup:")
print(f"  Total energy needed : {TOTAL_MWH} MWh")
print(f"  Time window         : {n_hours} hours (7 days)")
print(f"  Max per hour        : {MAX_PER_HOUR} MWh")
print(f"\n  Solving...")

result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

if result.success:
    optimal_schedule = result.x
    total_cost       = result.fun
    avg_price_paid   = total_cost / TOTAL_MWH

    # Naive strategy: buy evenly across all hours
    naive_schedule   = np.full(n_hours, TOTAL_MWH / n_hours)
    naive_cost       = np.dot(naive_schedule, predicted_prices)
    savings          = naive_cost - total_cost
    savings_pct      = (savings / naive_cost) * 100

    print(f"\n{'='*50}")
    print(f"  Optimization Results")
    print(f"{'='*50}")
    print(f"  Optimal total cost  : €{total_cost:,.2f}")
    print(f"  Naive total cost    : €{naive_cost:,.2f}")
    print(f"  💰 Savings          : €{savings:,.2f} ({savings_pct:.1f}%)")
    print(f"  Avg price paid      : €{avg_price_paid:.2f}/MWh")
    print(f"  Avg market price    : €{predicted_prices.mean():.2f}/MWh")
    print(f"{'='*50}")

    # Top 10 best hours to buy
    buy_hours = pd.DataFrame({
        "hour_index"  : range(n_hours),
        "datetime"    : week["time"].values,
        "predicted_price": predicted_prices,
        "buy_mwh"     : optimal_schedule
    })
    buying = buy_hours[buy_hours["buy_mwh"] > 0.01].sort_values("predicted_price")
    print(f"\n  Top 10 Cheapest Hours to Buy:")
    print(f"  {'DateTime':<30} {'Price':>10} {'Buy (MWh)':>10}")
    print(f"  {'-'*52}")
    for _, row in buying.head(10).iterrows():
        print(f"  {str(row['datetime']):<30} €{row['predicted_price']:>8.2f}  {row['buy_mwh']:>8.2f}")

    # ── Plot ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Top chart: predicted prices
    ax1.plot(predicted_prices, color="steelblue", label="Predicted Price")
    ax1.axhline(avg_price_paid, color="green",  linestyle="--", label=f"Avg Paid: €{avg_price_paid:.2f}")
    ax1.axhline(predicted_prices.mean(), color="red", linestyle="--", label=f"Market Avg: €{predicted_prices.mean():.2f}")
    ax1.set_title("Predicted Electricity Prices — Next 7 Days")
    ax1.set_ylabel("Price (€/MWh)")
    ax1.legend()

    # Bottom chart: optimal buy schedule
    colors = ["green" if x > 0.01 else "lightgray" for x in optimal_schedule]
    ax2.bar(range(n_hours), optimal_schedule, color=colors)
    ax2.set_title(f"Optimal Buy Schedule — Total {TOTAL_MWH} MWh | Savings: €{savings:,.2f} ({savings_pct:.1f}%)")
    ax2.set_ylabel("Buy Amount (MWh)")
    ax2.set_xlabel("Hour of Week")

    plt.tight_layout()
    plt.savefig("optimization.png")
    plt.show()

    # Save schedule
    buy_hours.to_csv("optimal_schedule.csv", index=False)
    print("\n✅ Optimal schedule saved to optimal_schedule.csv")
    print("🚀 Phase 4 Complete!")

else:
    print("❌ Optimization failed:", result.message)