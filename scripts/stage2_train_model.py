"""
Stage 2 -- Train RandomForest ML Model
Predicts: staff_needed per hour
Inputs:   data/qsr_sales_data.csv
Outputs:  models/staff_model.pkl
          reports/feature_importance.csv
          reports/model_summary.txt
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(BASE, "data",    "qsr_sales_data.csv")
MODEL_PATH   = os.path.join(BASE, "models",  "staff_model.pkl")
FI_PATH      = os.path.join(BASE, "reports", "feature_importance.csv")
SUMMARY_PATH = os.path.join(BASE, "reports", "model_summary.txt")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  {len(df):,} rows loaded")

# ── Feature engineering ───────────────────────────────────────────────────────
# Encode day_of_week as integer (0=Monday ... 6=Sunday)
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["day_of_week_num"] = df["day_of_week"].map({d: i for i, d in enumerate(DAY_ORDER)})

# Encode weather as integer
WEATHER_MAP = {"sunny": 0, "cloudy": 1, "rainy": 2, "snowy": 3}
df["weather_num"] = df["weather"].map(WEATHER_MAP)

# Features the scheduler knows IN ADVANCE when building next week's schedule
FEATURES = [
    "hour",             # hour of day (6-23)
    "day_of_week_num",  # 0=Monday, 6=Sunday
    "is_weekend",       # 1 if Sat or Sun
    "is_holiday",       # 1 if public holiday
    "is_promotion",     # 1 if promotion day
    "weather_num",      # 0=sunny, 1=cloudy, 2=rainy, 3=snowy
]

TARGET = "staff_needed"

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split (80/20, time-aware: test on last 20% of year) ──────────
# Sort by date + hour so the test set is the last ~73 days of 2024
df_sorted = df.sort_values(["date", "hour"]).reset_index(drop=True)
X_sorted = df_sorted[FEATURES]
y_sorted = df_sorted[TARGET]

split_idx = int(len(df_sorted) * 0.80)
X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

print(f"  Train size : {len(X_train):,} rows")
print(f"  Test size  : {len(X_test):,} rows")

# ── Train RandomForest ────────────────────────────────────────────────────────
print("\nTraining RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("  Training complete.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred_raw = model.predict(X_test)
y_pred = np.round(y_pred_raw).astype(int).clip(min=2)  # never predict below min staffing of 2

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"  MAE  (avg error in staff count) : {mae:.3f}")
print(f"  RMSE                            : {rmse:.3f}")
print(f"  R2 Score                        : {r2:.4f}")

# ── Feature importance ────────────────────────────────────────────────────────
fi = pd.DataFrame({
    "feature":   FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(f"\nFeature Importance:")
for _, row in fi.iterrows():
    bar = "#" * int(row["importance"] * 50)
    print(f"  {row['feature']:<20} {row['importance']:.4f}  {bar}")

# ── Sample predictions (show model reasoning) ─────────────────────────────────
print("\nSample Predictions vs Actual:")
sample = df_sorted.iloc[split_idx:split_idx+10][FEATURES + [TARGET]].copy()
sample["predicted"] = np.round(model.predict(sample[FEATURES])).astype(int).clip(min=2)
sample["hour_label"] = sample["hour"].apply(lambda h: f"{h:02d}:00")
sample["weather_label"] = sample["weather_num"].map({v: k for k, v in WEATHER_MAP.items()})
print(sample[["hour_label", "is_weekend", "weather_label", TARGET, "predicted"]].to_string(index=False))

# ── Save model ────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved --> {MODEL_PATH}")

# ── Save feature importance ───────────────────────────────────────────────────
fi.to_csv(FI_PATH, index=False)
print(f"Feature importance saved --> {FI_PATH}")

# ── Save summary report ───────────────────────────────────────────────────────
summary_lines = [
    "Smart Schedule Optimizer -- Stage 2 Model Summary",
    "=" * 50,
    f"Model     : RandomForestRegressor",
    f"Target    : staff_needed (rounded to nearest int, min 2)",
    f"Features  : {', '.join(FEATURES)}",
    f"Train rows: {len(X_train):,}",
    f"Test rows : {len(X_test):,}",
    "",
    "Performance",
    "-" * 30,
    f"MAE  : {mae:.3f}  (avg off by {mae:.2f} staff members)",
    f"RMSE : {rmse:.3f}",
    f"R2   : {r2:.4f}",
    "",
    "Feature Importance",
    "-" * 30,
]
for _, row in fi.iterrows():
    summary_lines.append(f"  {row['feature']:<20} {row['importance']:.4f}")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Summary report saved --> {SUMMARY_PATH}")

print("\nStage 2 complete. Model ready for Stage 3 (Optimizer).")
