"""
Stage 2b -- Enhanced ML Model with LY context, trend, and reliability scoring
Trains on 2024 data using 2023 (LY) as reference features.

New features vs Stage 2:
  ly_same_week_cust      -- avg customer_count, same ISO week + hour in LY (2023)
  ly_same_week_txn       -- avg transaction_count, same ISO week + hour in LY
  ly_context_cust        -- avg customers, ISO week +/-3 around same week in LY
  ly_context_txn         -- avg transactions, ISO week +/-3 around same week in LY
  trend_ratio            -- avg customers last 3 weeks / avg prior 3 weeks (same hour)

Outputs:
  models/staff_model_enhanced.pkl
  reports/enhanced_model_summary.txt
  reports/feature_importance_enhanced.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE          = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH     = os.path.join(BASE, "data",    "qsr_sales_data.csv")
MODEL_PATH    = os.path.join(BASE, "models",  "staff_model_enhanced.pkl")
FI_PATH       = os.path.join(BASE, "reports", "feature_importance_enhanced.csv")
SUMMARY_PATH  = os.path.join(BASE, "reports", "enhanced_model_summary.txt")

DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEATHER_MAP = {"sunny": 0, "cloudy": 1, "rainy": 2, "snowy": 3}

# ── Load & split years ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Total rows: {len(df):,}")

df_2023 = df[df["year"] == 2023].copy()
df_2024 = df[df["year"] == 2024].copy()
print(f"  2023 rows: {len(df_2023):,}  |  2024 rows: {len(df_2024):,}")

# ── Precompute LY (2023) weekly-hourly aggregates ─────────────────────────────
# Group by (iso_week, hour) to get average customers and transactions per slot in 2023
ly_agg = (df_2023
    .groupby(["iso_week", "hour"])
    .agg(
        ly_cust_mean=("customer_count",    "mean"),
        ly_cust_std=("customer_count",     "std"),
        ly_txn_mean=("transaction_count",  "mean"),
    )
    .reset_index()
)
ly_agg["ly_cust_std"] = ly_agg["ly_cust_std"].fillna(0)

def get_ly_same_week(iso_week: int, hour: int) -> tuple:
    """Returns (avg_customers, avg_transactions) for same iso_week + hour in 2023."""
    row = ly_agg[(ly_agg["iso_week"] == iso_week) & (ly_agg["hour"] == hour)]
    if row.empty:
        # Fallback: use adjacent week
        nearby = ly_agg[
            (ly_agg["iso_week"].between(iso_week - 1, iso_week + 1)) &
            (ly_agg["hour"] == hour)
        ]
        if nearby.empty:
            return (0.0, 0.0, 0.0)
        return (nearby["ly_cust_mean"].mean(), nearby["ly_txn_mean"].mean(),
                nearby["ly_cust_std"].mean())
    return (float(row["ly_cust_mean"].iloc[0]),
            float(row["ly_txn_mean"].iloc[0]),
            float(row["ly_cust_std"].iloc[0]))

def get_ly_context(iso_week: int, hour: int, window: int = 3) -> tuple:
    """Returns (avg_customers, avg_transactions) for iso_week +/-window in 2023."""
    weeks = list(range(max(1, iso_week - window), min(53, iso_week + window + 1)))
    ctx = ly_agg[
        (ly_agg["iso_week"].isin(weeks)) &
        (ly_agg["hour"] == hour)
    ]
    if ctx.empty:
        return (0.0, 0.0)
    return (ctx["ly_cust_mean"].mean(), ctx["ly_txn_mean"].mean())

# ── Precompute 2024 weekly-hourly averages for trend calculation ──────────────
# Used to compute: last 3 weeks avg / prior 3 weeks avg (same hour)
weekly_hourly_2024 = (df_2024
    .groupby(["iso_week", "hour"])["customer_count"]
    .mean()
    .reset_index()
    .rename(columns={"customer_count": "weekly_avg"})
)

def get_trend_ratio(iso_week: int, hour: int) -> float:
    """
    Trend ratio = avg(customer_count last 3 weeks) / avg(customer_count prior 3 weeks)
    For weeks 1-3, falls back to using 2023 LY weeks.
    > 1.0 means growing, < 1.0 means declining.
    """
    recent_weeks = list(range(max(1, iso_week - 3), iso_week))
    prior_weeks  = list(range(max(1, iso_week - 6), max(1, iso_week - 3)))

    # Try 2024 data first
    def week_avg(weeks, source_df):
        rows = source_df[
            (source_df["iso_week"].isin(weeks)) &
            (source_df["hour"] == hour)
        ]
        return rows["weekly_avg"].mean() if not rows.empty else None

    recent_avg = week_avg(recent_weeks, weekly_hourly_2024)
    prior_avg  = week_avg(prior_weeks,  weekly_hourly_2024)

    # Fall back to LY if 2024 weeks not available yet (early year)
    if recent_avg is None or np.isnan(recent_avg):
        r = ly_agg[(ly_agg["iso_week"].isin(recent_weeks)) & (ly_agg["hour"] == hour)]
        recent_avg = r["ly_cust_mean"].mean() if not r.empty else None
    if prior_avg is None or np.isnan(prior_avg):
        p = ly_agg[(ly_agg["iso_week"].isin(prior_weeks)) & (ly_agg["hour"] == hour)]
        prior_avg = p["ly_cust_mean"].mean() if not p.empty else None

    if prior_avg and prior_avg > 0 and recent_avg:
        return round(float(recent_avg) / float(prior_avg), 4)
    return 1.0  # neutral: no trend data available

# ── Feature engineering for 2024 training data ───────────────────────────────
print("\nEngineering enhanced features for 2024 training data...")

df_2024 = df_2024.copy()
df_2024["day_of_week_num"] = df_2024["day_of_week"].map({d: i for i, d in enumerate(DAY_ORDER)})
df_2024["weather_num"]     = df_2024["weather"].map(WEATHER_MAP)

# LY same-week features
print("  Computing LY same-week features...")
ly_same = df_2024.apply(
    lambda r: pd.Series(
        get_ly_same_week(int(r["iso_week"]), int(r["hour"])),
        index=["ly_same_week_cust", "ly_same_week_txn", "ly_cust_variance"]
    ), axis=1
)
df_2024 = pd.concat([df_2024, ly_same], axis=1)

# LY context (+/-3 weeks) features
print("  Computing LY context (+/-3 week) features...")
ly_ctx = df_2024.apply(
    lambda r: pd.Series(
        get_ly_context(int(r["iso_week"]), int(r["hour"])),
        index=["ly_context_cust", "ly_context_txn"]
    ), axis=1
)
df_2024 = pd.concat([df_2024, ly_ctx], axis=1)

# Trend ratio (recent 3 weeks / prior 3 weeks)
print("  Computing trend ratios...")
df_2024["trend_ratio"] = df_2024.apply(
    lambda r: get_trend_ratio(int(r["iso_week"]), int(r["hour"])), axis=1
)

# ── Features & target ─────────────────────────────────────────────────────────
FEATURES = [
    # Original time/event features
    "hour",
    "day_of_week_num",
    "is_weekend",
    "is_holiday",
    "is_promotion",
    "weather_num",
    # New LY features (same week)
    "ly_same_week_cust",
    "ly_same_week_txn",
    # New LY context features (+/-3 weeks)
    "ly_context_cust",
    "ly_context_txn",
    # Trend
    "trend_ratio",
]
TARGET = "staff_needed"

# ── Time-aware train/test split (last 20% of 2024 = ~73 days as test) ─────────
df_2024_sorted = df_2024.sort_values(["date", "hour"]).reset_index(drop=True)
split_idx = int(len(df_2024_sorted) * 0.80)

X_train = df_2024_sorted.iloc[:split_idx][FEATURES]
X_test  = df_2024_sorted.iloc[split_idx:][FEATURES]
y_train = df_2024_sorted.iloc[:split_idx][TARGET]
y_test  = df_2024_sorted.iloc[split_idx:][TARGET]

print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# ── Train enhanced RandomForest ───────────────────────────────────────────────
print("\nTraining enhanced RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("  Training complete.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = np.round(model.predict(X_test)).astype(int).clip(min=2)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = mean_squared_error(y_test, y_pred) ** 0.5
r2     = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set (2024 holdout):")
print(f"  MAE  : {mae:.3f}  (avg error: {mae:.2f} staff)")
print(f"  RMSE : {rmse:.3f}")
print(f"  R2   : {r2:.4f}")

# ── Feature importance ────────────────────────────────────────────────────────
fi = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(f"\nFeature Importance:")
for _, row in fi.iterrows():
    bar = "#" * int(row["importance"] * 50)
    print(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")

# ── Reliability scoring function ──────────────────────────────────────────────
def reliability_score(iso_week: int, hour: int, weeks_ahead: int) -> dict:
    """
    Computes how reliable a forward projection is.

    Factors:
    - LY variance: high variance in same-week LY data = lower confidence
    - weeks_ahead: each week further ahead reduces confidence by ~3%
    - trend_ratio deviation from 1.0: strong trend (up or down) = slightly less certain

    Returns a dict with score (0-1), percent string, and label.
    """
    _, _, ly_std = get_ly_same_week(iso_week, hour)
    ly_mean, _   = get_ly_context(iso_week, hour)

    # Coefficient of variation (how volatile is this week historically)
    cv = (ly_std / ly_mean) if ly_mean > 0 else 0.15
    cv = min(cv, 0.40)  # cap at 40%

    # Base reliability from variance (low variance = high reliability)
    base = 1.0 - (cv * 1.5)  # e.g. cv=0.10 -> base=0.85
    base = max(0.50, min(0.95, base))

    # Decay per week ahead (3% per week, based on typical QSR forecast error)
    decay = 0.97 ** weeks_ahead

    score = round(base * decay, 3)
    pct   = f"{score * 100:.0f}%"

    if score >= 0.85:
        label = "HIGH"
    elif score >= 0.72:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {"score": score, "pct": pct, "label": label}

# ── Save model, feature importance, summary ───────────────────────────────────
# Save feature list alongside model so stage3 can load it
model_bundle = {
    "model":    model,
    "features": FEATURES,
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)
print(f"\nModel saved --> {MODEL_PATH}")

fi.to_csv(FI_PATH, index=False)
print(f"Feature importance saved --> {FI_PATH}")

summary_lines = [
    "Smart Schedule Optimizer -- Stage 2b Enhanced Model Summary",
    "=" * 60,
    f"Model     : RandomForestRegressor (300 trees, depth 12)",
    f"Target    : staff_needed (rounded, min 2)",
    f"Train yr  : 2024  |  LY reference: 2023",
    f"Features  : {len(FEATURES)} total",
    "",
    "Performance on 2024 holdout (last 20% of year)",
    "-" * 40,
    f"MAE  : {mae:.3f}  (avg off by {mae:.2f} staff members)",
    f"RMSE : {rmse:.3f}",
    f"R2   : {r2:.4f}",
    "",
    "Feature Importance",
    "-" * 40,
]
for _, row in fi.iterrows():
    summary_lines.append(f"  {row['feature']:<25} {row['importance']:.4f}")

summary_lines += [
    "",
    "Reliability Scoring Logic",
    "-" * 40,
    "  base_score = 1.0 - (LY_cv * 1.5)     [LY_cv = std/mean of same week LY]",
    "  base_score clamped to [0.50, 0.95]",
    "  reliability = base_score * 0.97^weeks_ahead",
    "  HIGH >= 85% | MEDIUM >= 72% | LOW < 72%",
    "",
    "Growth Factor (applied in Stage 3 optimizer)",
    "-" * 40,
    "  Projected = ML_prediction * 1.05  (5% YoY growth)",
    "  Applied to staff_needed, customer projection, transaction projection",
]

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Summary saved --> {SUMMARY_PATH}")

# ── Save LY aggregates for Stage 3 (optimizer needs LY lookups at runtime) ───
ly_agg_path = os.path.join(BASE, "models", "ly_agg_2023.csv")
ly_agg.to_csv(ly_agg_path, index=False)

weekly_hourly_path = os.path.join(BASE, "models", "weekly_hourly_2024.csv")
weekly_hourly_2024.to_csv(weekly_hourly_path, index=False)
print(f"LY aggregates saved --> {ly_agg_path}")
print(f"2024 weekly hourly averages saved --> {weekly_hourly_path}")

print("\nStage 2b complete. Enhanced model ready for Stage 3.")
