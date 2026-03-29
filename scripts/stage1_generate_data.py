"""
Stage 1 -- Generate QSR Sales Data (2 years: 2023 + 2024)
Outputs: data/qsr_sales_data.csv
Records: 13,158 rows (731 days x 18 hours)
- 2024 = base year (5% more than 2023, reflecting YoY growth)
- 2023 = last year (LY) reference data
New columns vs original: transaction_count, iso_week, year
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# ── Alberta/Canadian holidays per year ───────────────────────────────────────
HOLIDAYS = {
    2023: {
        date(2023, 1, 1),   # New Year's Day
        date(2023, 2, 20),  # Family Day (Alberta) -- 3rd Monday of Feb
        date(2023, 4, 7),   # Good Friday
        date(2023, 4, 10),  # Easter Monday
        date(2023, 5, 22),  # Victoria Day
        date(2023, 7, 1),   # Canada Day
        date(2023, 8, 7),   # Heritage Day (Alberta) -- 1st Monday of Aug
        date(2023, 9, 4),   # Labour Day
        date(2023, 10, 9),  # Thanksgiving
        date(2023, 11, 11), # Remembrance Day
        date(2023, 12, 25), # Christmas Day
        date(2023, 12, 26), # Boxing Day
    },
    2024: {
        date(2024, 1, 1),
        date(2024, 2, 19),  # Family Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 4, 1),   # Easter Monday
        date(2024, 5, 20),  # Victoria Day
        date(2024, 7, 1),   # Canada Day
        date(2024, 8, 5),   # Heritage Day
        date(2024, 9, 2),   # Labour Day
        date(2024, 10, 14), # Thanksgiving
        date(2024, 11, 11), # Remembrance Day
        date(2024, 12, 25), # Christmas Day
        date(2024, 12, 26), # Boxing Day
    },
}

# ── Hourly base traffic (customers) ──────────────────────────────────────────
HOURLY_BASE = {
    6: 15,  7: 35,  8: 45,  9: 35,  10: 28, 11: 50,
    12: 80, 13: 65, 14: 30, 15: 28, 16: 32, 17: 55,
    18: 80, 19: 60, 20: 40, 21: 30, 22: 20, 23: 12,
}

# Peak hours have more group transactions (lower txn/customer ratio)
HOURLY_TXN_RATIO = {
    6: 0.96, 7: 0.95, 8: 0.94, 9: 0.95, 10: 0.95, 11: 0.92,
    12: 0.88, 13: 0.90, 14: 0.94, 15: 0.95, 16: 0.95, 17: 0.91,
    18: 0.88, 19: 0.90, 20: 0.93, 21: 0.95, 22: 0.96, 23: 0.97,
}

WEATHER_OPTIONS = ["sunny", "cloudy", "rainy", "snowy"]
WEATHER_MULT    = {"sunny": 1.00, "cloudy": 0.95, "rainy": 0.85, "snowy": 0.70}

# YoY growth: 2024 is 5% more than 2023
YEAR_SCALE = {2023: 1 / 1.05, 2024: 1.0}


def get_weather(d: date, rng: np.random.Generator) -> str:
    month = d.month
    if month in [12, 1, 2]:
        probs = [0.15, 0.30, 0.05, 0.50]
    elif month in [3, 4]:
        probs = [0.25, 0.30, 0.25, 0.20]
    elif month in [5, 6, 7, 8]:
        probs = [0.50, 0.30, 0.18, 0.02]
    else:
        probs = [0.25, 0.35, 0.25, 0.15]
    return rng.choice(WEATHER_OPTIONS, p=probs)


def build_year(year: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    holidays = HOLIDAYS[year]
    scale    = YEAR_SCALE[year]

    # Days in year
    start = date(year, 1, 1)
    days  = 366 if year % 4 == 0 else 365
    all_dates = [start + timedelta(days=i) for i in range(days)]

    # Promotion days (~10% of year)
    promo_flags = {d: int(rng.random() < 0.10) for d in all_dates}

    # Daily weather
    daily_weather = {d: get_weather(d, rng) for d in all_dates}

    rows = []
    for d in all_dates:
        weather    = daily_weather[d]
        is_holiday = 1 if d in holidays else 0
        is_promo   = promo_flags[d]
        is_weekend = 1 if d.weekday() >= 5 else 0
        day_name   = d.strftime("%A")
        iso_week   = d.isocalendar()[1]

        for hour in range(6, 24):
            base = HOURLY_BASE[hour]

            mult = scale
            if is_weekend: mult *= 1.30
            if is_holiday: mult *= 1.50
            if is_promo:   mult *= 1.20
            mult *= WEATHER_MULT[weather]

            noise     = rng.uniform(0.85, 1.15)
            customers = max(0, int(base * mult * noise))

            # Transaction count: slightly lower than customers (group orders at peaks)
            txn_ratio    = HOURLY_TXN_RATIO[hour] * rng.uniform(0.97, 1.03)
            transactions = max(1 if customers > 0 else 0,
                               int(customers * txn_ratio))

            sales = round(customers * 12.50 * rng.uniform(0.95, 1.05), 2)

            # Staff needed: 1 per 15 customers, min 2, +1 at peak if busy
            staff = max(2, int(np.ceil(customers / 15)))
            if hour in [12, 18] and customers >= 60:
                staff += 1

            rows.append({
                "year":              year,
                "iso_week":          iso_week,
                "date":              d.strftime("%Y-%m-%d"),
                "hour":              hour,
                "day_of_week":       day_name,
                "is_weekend":        is_weekend,
                "is_holiday":        is_holiday,
                "is_promotion":      is_promo,
                "weather":           weather,
                "customer_count":    customers,
                "transaction_count": transactions,
                "sales_amount":      sales,
                "staff_needed":      staff,
            })
    return rows

# ── Generate both years ───────────────────────────────────────────────────────
print("Generating 2023 data (LY baseline)...")
rows_2023 = build_year(2023, seed=7)
print(f"  {len(rows_2023):,} rows")

print("Generating 2024 data (current year, +5% YoY)...")
rows_2024 = build_year(2024, seed=42)
print(f"  {len(rows_2024):,} rows")

df = pd.DataFrame(rows_2023 + rows_2024)
df = df.sort_values(["date", "hour"]).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(BASE, "data", "qsr_sales_data.csv")
df.to_csv(out, index=False)

print(f"\nDONE: qsr_sales_data.csv saved -- {len(df):,} total rows")
print(f"   Date range : {df['date'].min()} to {df['date'].max()}")
print()

for yr in [2023, 2024]:
    yr_df = df[df["year"] == yr]
    print(f"   {yr}  customers/hr avg: {yr_df['customer_count'].mean():.1f}"
          f"  txn/hr avg: {yr_df['transaction_count'].mean():.1f}"
          f"  sales/hr avg: ${yr_df['sales_amount'].mean():.2f}")

growth = (df[df["year"]==2024]["customer_count"].mean() /
          df[df["year"]==2023]["customer_count"].mean() - 1) * 100
print(f"\n   YoY growth (customers): +{growth:.1f}%  (target: +5.0%)")
