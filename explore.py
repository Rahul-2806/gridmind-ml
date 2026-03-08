import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────
energy = pd.read_csv("energy_dataset.csv")
energy["time"] = pd.to_datetime(energy["time"], utc=True)
energy = energy.set_index("time")

weather = pd.read_csv("weather_features.csv")
weather["dt_iso"] = pd.to_datetime(weather["dt_iso"], utc=True)
weather = weather.set_index("dt_iso")

print("✅ Energy dataset shape:", energy.shape)
print("✅ Weather dataset shape:", weather.shape)
print("\n── Energy Columns ──")
print(energy.columns.tolist())

# ── Check missing values ───────────────────────────────
print("\n── Missing Values (Energy) ──")
print(energy.isnull().sum()[energy.isnull().sum() > 0])

# ── Our target variable ────────────────────────────────
# "price actual" = the real hourly electricity price in €/MWh
print("\n── Price Statistics ──")
print(energy["price actual"].describe())

# ── Plot price history ─────────────────────────────────
plt.figure(figsize=(15, 4))
energy["price actual"].plot(title="Hourly Electricity Prices (€/MWh)")
plt.ylabel("Price (€/MWh)")
plt.tight_layout()
plt.savefig("price_history.png")
plt.show()

# ── Feature Engineering ────────────────────────────────
df = energy[["price actual"]].copy()

# Time features
df["hour"]       = df.index.hour
df["dayofweek"]  = df.index.dayofweek   # 0=Monday, 6=Sunday
df["month"]      = df.index.month
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

# Lag features (past prices as input to predict future)
df["price_lag_1h"]  = df["price actual"].shift(1)   # 1 hour ago
df["price_lag_24h"] = df["price actual"].shift(24)  # same hour yesterday
df["price_lag_168h"]= df["price actual"].shift(168) # same hour last week

# Rolling averages
df["rolling_mean_24h"] = df["price actual"].rolling(24).mean()
df["rolling_std_24h"]  = df["price actual"].rolling(24).std()

# Drop rows with NaN from lag creation
df.dropna(inplace=True)

print("\n── Feature Engineered Dataset ──")
print(df.head())
print(f"\nFinal shape: {df.shape}")

# ── Correlation heatmap ────────────────────────────────
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation with Price")
plt.tight_layout()
plt.savefig("correlation.png")
plt.show()

# Save processed data
df.to_csv("processed_data.csv")
print("\n✅ Processed data saved to processed_data.csv")