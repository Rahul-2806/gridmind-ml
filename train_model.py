import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pickle

# ── Load & Prepare Data ────────────────────────────────
energy = pd.read_csv("energy_dataset.csv")
energy["time"] = pd.to_datetime(energy["time"], utc=True)
energy = energy.set_index("time")

df = energy[["price actual"]].copy()

# Fill small missing values
df["price actual"] = df["price actual"].fillna(method="ffill")

# ── Feature Engineering ────────────────────────────────
df["hour"]        = df.index.hour
df["dayofweek"]   = df.index.dayofweek
df["month"]       = df.index.month
df["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)

# Lag features
df["price_lag_1h"]   = df["price actual"].shift(1)
df["price_lag_24h"]  = df["price actual"].shift(24)
df["price_lag_168h"] = df["price actual"].shift(168)

# Rolling features
df["rolling_mean_24h"] = df["price actual"].rolling(24).mean()
df["rolling_std_24h"]  = df["price actual"].rolling(24).std()
df["rolling_mean_7d"]  = df["price actual"].rolling(168).mean()

df.dropna(inplace=True)

# ── Split Features & Target ────────────────────────────
FEATURES = [
    "hour", "dayofweek", "month", "is_weekend",
    "price_lag_1h", "price_lag_24h", "price_lag_168h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_7d"
]

X = df[FEATURES]
y = df["price actual"]

# Time-based split (don't shuffle — this is time series!)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Train XGBoost Model ────────────────────────────────
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)],
          verbose=100)

# ── Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  Model Performance")
print(f"{'='*40}")
print(f"  MAE  : €{mae:.2f}/MWh")
print(f"  RMSE : €{rmse:.2f}/MWh")
print(f"  R²   : {r2:.4f}  (1.0 = perfect)")
print(f"{'='*40}")

# ── Plot Predictions vs Actual ─────────────────────────
plt.figure(figsize=(15, 5))
plt.plot(y_test.values[-200:], label="Actual Price", color="blue", alpha=0.7)
plt.plot(y_pred[-200:],        label="Predicted Price", color="red", alpha=0.7)
plt.title("Actual vs Predicted Electricity Prices (last 200 hours)")
plt.ylabel("Price (€/MWh)")
plt.xlabel("Hours")
plt.legend()
plt.tight_layout()
plt.savefig("predictions.png")
plt.show()

# ── Feature Importance ─────────────────────────────────
plt.figure(figsize=(10, 5))
pd.Series(model.feature_importances_, index=FEATURES).sort_values().plot(
    kind="barh", color="steelblue", title="Feature Importance"
)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ── Save Model ─────────────────────────────────────────
with open("gridmind_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved to gridmind_model.pkl")
print("🚀 Phase 2 Complete!")