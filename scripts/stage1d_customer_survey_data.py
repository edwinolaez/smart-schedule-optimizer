"""
Stage 1d — Generate Customer Survey Data
=========================================
WHY THIS MATTERS (Business Reason):
  The central theory of this platform is:
    Better scheduling → higher staff morale → better customer experience
    → consistent sales growth.

  Right now we can PREDICT how many staff are needed (Module 1 ML model).
  But we can't yet prove the link between staffing quality and how customers
  actually felt about their visit.

  This dataset is the "customer side of the story." It captures how guests
  rated their experience — speed, friendliness, accuracy, cleanliness,
  product availability — so we can later correlate those scores with:
    - Whether the store was adequately staffed that hour
    - Whether a supervisor was on the floor (Morale Factor 3)
    - Whether it was a high-traffic promotional day (stock runout risk)

WHAT THIS SCRIPT DOES:
  1. Loads qsr_sales_data.csv to get real day context (weather, customer
     count, holidays, promotions) for every date in 2024
  2. Generates ~5 survey responses per day = ~1,830 rows total
  3. Makes ratings realistic by adjusting scores based on conditions:
       - Peak hours (12pm, 6pm): speed of service drops slightly
       - High customer volume: speed drops more
       - Holidays/promotions: product availability at risk
       - Bad weather: cleanliness score dips
       - Weekends: friendliness and speed both take a small hit
  4. Calculates CSAT score (average of the 5 core ratings)
  5. Calculates NPS flag (Promoter / Passive / Detractor)
  6. Saves to data/qsr_customer_survey.csv

THE 16 COLUMNS:
  survey_id            — unique ID for each response (S0001, S0002, ...)
  date                 — date of the visit (YYYY-MM-DD)
  iso_week             — ISO week number (for trend analysis)
  day_of_week          — Monday, Tuesday, etc.
  is_weekend           — 1 if Saturday or Sunday, else 0
  visit_hour           — hour the customer visited (6-22)
  shift_period         — morning (6-11) / afternoon (12-16) / evening (17-22)
  speed_of_service     — rating 1-5 (was their order fast?)
  staff_friendliness   — rating 1-5 (were staff welcoming?)
  order_accuracy       — rating 1-5 (did they get what they ordered?)
  cleanliness          — rating 1-5 (was the dining area clean?)
  product_availability — rating 1-5 (was everything on the menu available?)
  overall_satisfaction — rating 1-5 (overall, how was the experience?)
  likelihood_to_return — rating 0-10 (NPS question: would they recommend?)
  csat_score           — float, average of the 5 core ratings (1.0-5.0)
  nps_flag             — Promoter (9-10) / Passive (7-8) / Detractor (0-6)

Outputs:
  data/qsr_customer_survey.csv   (~1,830 rows, 16 columns)
"""

import pandas as pd    # our spreadsheet tool
import numpy as np     # math and random number generation
import os              # file path builder

# ── Random seed ──────────────────────────────────────────────────────────────
# Think of this like setting a fixed shuffle before a card game.
# It means every time we run this script we get the SAME "random" data.
# That's important for reproducibility — your portfolio reviewer can run
# the script and get the exact same dataset you built.
np.random.seed(42)

# ── File paths ────────────────────────────────────────────────────────────────
BASE        = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SALES_PATH  = os.path.join(BASE, "data", "qsr_sales_data.csv")
OUTPUT_PATH = os.path.join(BASE, "data", "qsr_customer_survey.csv")

# ── Load sales data to use as day-level context ───────────────────────────────
# We only want 2024 — that's the year our ML model is trained on.
# The sales data gives us the "conditions" for each hour:
# weather, customer volume, holidays, promotions.
# Think of it as pulling your daily sales report to understand what kind of
# day it was before reading the comment cards from that day.
print("Loading sales data for 2024 context...")
sales = pd.read_csv(SALES_PATH)
sales = sales[sales["year"] == 2024].copy()   # filter to 2024 only
print(f"  {len(sales):,} hourly rows loaded for 2024")

# ── Build a day-level summary from the hourly sales data ─────────────────────
# We collapse from hourly rows to one row per day.
# For each date we want to know:
#   - the weather that day
#   - whether it was a holiday
#   - whether there was a promotion
#   - peak customer count (busiest hour)
#   - iso_week, day_of_week, is_weekend
#
# Think of this like going from your hourly transaction tape
# to your end-of-day summary sheet.
print("Building daily context from sales data...")
day_context = (
    sales.groupby("date")
    .agg(
        iso_week          = ("iso_week",        "first"),
        day_of_week       = ("day_of_week",     "first"),
        is_weekend        = ("is_weekend",      "first"),
        is_holiday        = ("is_holiday",      "first"),
        is_promotion      = ("is_promotion",    "first"),
        weather           = ("weather",         "first"),
        peak_customers    = ("customer_count",  "max"),    # busiest hour that day
    )
    .reset_index()
)
print(f"  {len(day_context)} unique days in 2024")

# ── Define how visit hours are distributed across the day ──────────────────
# Not all hours get equal foot traffic. We weight survey submissions
# toward busy times — just like real comment cards pile up during lunch
# and dinner rushes.
#
# The store is open 6am-10pm (last visit hour = 22).
# Weights = relative likelihood of a survey coming from each hour.
VISIT_HOURS = list(range(6, 23))   # hours 6 through 22 inclusive

# Higher weight = more surveys from that hour
# 12 (lunch) and 18 (dinner) are the peaks
HOUR_WEIGHTS = {
     6: 1,  7: 2,  8: 3,  9: 3, 10: 3,
    11: 4, 12: 7, 13: 6, 14: 4, 15: 3,
    16: 3, 17: 4, 18: 7, 19: 6, 20: 4,
    21: 3, 22: 2,
}
# Convert to a probability list so numpy can sample from it
weight_list  = [HOUR_WEIGHTS[h] for h in VISIT_HOURS]
total_weight = sum(weight_list)
hour_probs   = [w / total_weight for w in weight_list]  # must sum to 1.0

# ── Shift period label based on visit hour ────────────────────────────────
# In QSR we think in shifts, not clock hours.
# This lets Module 2 (Training) correlate survey results with which
# shift team was on duty.
def get_shift_period(hour: int) -> str:
    if hour <= 11:
        return "morning"     # 6am-11am  — opening crew
    elif hour <= 16:
        return "afternoon"   # 12pm-4pm  — lunch crew
    else:
        return "evening"     # 5pm-10pm  — closing crew

# ── Base ratings: what a "normal" day looks like ──────────────────────────
# These are the average scores a well-run QSR earns on a typical day.
# 4.0/5.0 = "good but not perfect" — realistic for a busy fast-food location.
BASE_SPEED        = 3.90   # speed of service is hardest to maintain
BASE_FRIENDLINESS = 4.10   # staff are generally upbeat
BASE_ACCURACY     = 4.05   # order accuracy is usually high
BASE_CLEANLINESS  = 4.00   # cleanliness is consistent but slips at peaks
BASE_AVAIL        = 4.20   # product availability is usually reliable

# ── Generate survey rows ──────────────────────────────────────────────────
# We loop through every day in 2024, and for each day we generate
# SURVEYS_PER_DAY individual customer responses.
# Think of this like pulling comment cards from the box at the end of each day.
SURVEYS_PER_DAY = 5   # ~5 customers per day leave a review

print(f"Generating ~{len(day_context) * SURVEYS_PER_DAY:,} survey responses...")

rows = []          # this list will collect every survey response as a dict
survey_counter = 0 # we'll use this to build the survey_id (S0001, S0002, ...)

for _, day in day_context.iterrows():

    # ── Pull today's context ──────────────────────────────────────────────
    is_wknd    = day["is_weekend"]
    is_hol     = day["is_holiday"]
    is_promo   = day["is_promotion"]
    weather    = day["weather"]
    peak_cust  = day["peak_customers"]

    for _ in range(SURVEYS_PER_DAY):

        # Pick a random visit hour weighted by foot traffic
        visit_hour = int(np.random.choice(VISIT_HOURS, p=hour_probs))

        # ── Start with base scores ────────────────────────────────────────
        speed   = BASE_SPEED
        friend  = BASE_FRIENDLINESS
        accur   = BASE_ACCURACY
        clean   = BASE_CLEANLINESS
        avail   = BASE_AVAIL

        # ── Adjust for peak hour pressure ─────────────────────────────────
        # At 12pm and 6pm, lines are long and staff are stretched thin.
        # Speed takes the biggest hit. Friendliness also dips slightly —
        # a stressed crew member is less likely to smile and upsell.
        if visit_hour in (12, 13, 18, 19):
            speed  -= 0.40
            friend -= 0.20

        # ── Adjust for high customer volume ───────────────────────────────
        # The busier the store, the harder it is to maintain speed.
        # We measure pressure as how far above "comfortable" (60 cust/hr)
        # the peak traffic was.
        volume_pressure = max(0, (peak_cust - 60) / 100)   # ranges 0.0 - ~0.3
        speed  -= volume_pressure
        accur  -= volume_pressure * 0.5   # accuracy also drops under pressure

        # ── Adjust for weekend ────────────────────────────────────────────
        # Weekends = more families, more complex orders, higher stress.
        if is_wknd:
            speed  -= 0.20
            friend -= 0.10

        # ── Adjust for holidays ───────────────────────────────────────────
        # Holidays are our busiest days. Product availability takes a hit
        # (stock runouts), and speed suffers with the volume spike.
        if is_hol:
            speed  -= 0.30
            avail  -= 0.35

        # ── Adjust for promotional days ───────────────────────────────────
        # A promo brings in more customers. If we weren't adequately staffed,
        # speed drops and we risk running out of promo items.
        if is_promo:
            speed  -= 0.20
            avail  -= 0.25

        # ── Adjust for bad weather ────────────────────────────────────────
        # Rainy days = wet shoes tracked in = harder to keep floor clean.
        # Snowy days = even worse. Staff are also busier with outdoor cleanup.
        if weather == "rainy":
            clean -= 0.20
        elif weather == "snowy":
            clean -= 0.35

        # ── Add realistic randomness ──────────────────────────────────────
        # No two customers experience a visit the same way.
        # We add small random noise (like rolling a die) to each score.
        # std=0.45 means most noise falls within ±0.9 of the adjusted base.
        noise = np.random.normal(0, 0.45, 5)
        speed  += noise[0]
        friend += noise[1]
        accur  += noise[2]
        clean  += noise[3]
        avail  += noise[4]

        # ── Clip all scores to valid range 1.0–5.0 ───────────────────────
        # Ratings can't go below 1 or above 5 — like a star review system.
        speed  = float(np.clip(round(speed,  1), 1.0, 5.0))
        friend = float(np.clip(round(friend, 1), 1.0, 5.0))
        accur  = float(np.clip(round(accur,  1), 1.0, 5.0))
        clean  = float(np.clip(round(clean,  1), 1.0, 5.0))
        avail  = float(np.clip(round(avail,  1), 1.0, 5.0))

        # ── Overall satisfaction ──────────────────────────────────────────
        # Customers weigh speed and friendliness more than cleanliness
        # when forming their overall impression. This weighted average
        # reflects how QSR guest satisfaction research is typically scored.
        overall = (
            speed   * 0.25 +
            friend  * 0.25 +
            accur   * 0.20 +
            clean   * 0.15 +
            avail   * 0.15
        )
        overall += np.random.normal(0, 0.20)   # a little extra personal variation
        overall  = float(np.clip(round(overall, 1), 1.0, 5.0))

        # ── CSAT score ────────────────────────────────────────────────────
        # Customer Satisfaction Score = simple average of all 5 core ratings.
        # Used for tracking trends month over month.
        csat = round((speed + friend + accur + clean + avail) / 5, 2)

        # ── Likelihood to Return (NPS question) ───────────────────────────
        # "On a scale of 0-10, how likely are you to recommend us?"
        # We derive this from overall satisfaction so they're correlated.
        # A 5/5 overall → likely 9 or 10 on NPS scale.
        # A 2/5 overall → likely 2 or 3 on NPS scale.
        ltr_base = round((overall - 1) / 4 * 10)   # scale 1-5 → 0-10
        ltr_noise = int(np.random.choice([-1, 0, 0, 1, 1], p=[0.1, 0.3, 0.3, 0.2, 0.1]))
        ltr = int(np.clip(ltr_base + ltr_noise, 0, 10))

        # ── NPS flag ──────────────────────────────────────────────────────
        # Standard Net Promoter Score buckets used industry-wide:
        #   9-10 = Promoter   → loyal fans who will refer others
        #   7-8  = Passive     → satisfied but not enthusiastic
        #   0-6  = Detractor   → unhappy, may warn others away
        if ltr >= 9:
            nps_flag = "Promoter"
        elif ltr >= 7:
            nps_flag = "Passive"
        else:
            nps_flag = "Detractor"

        # ── Build the row ─────────────────────────────────────────────────
        survey_counter += 1
        rows.append({
            "survey_id":            f"S{survey_counter:04d}",
            "date":                 day["date"],
            "iso_week":             int(day["iso_week"]),
            "day_of_week":          day["day_of_week"],
            "is_weekend":           int(is_wknd),
            "visit_hour":           visit_hour,
            "shift_period":         get_shift_period(visit_hour),
            "speed_of_service":     speed,
            "staff_friendliness":   friend,
            "order_accuracy":       accur,
            "cleanliness":          clean,
            "product_availability": avail,
            "overall_satisfaction": overall,
            "likelihood_to_return": ltr,
            "csat_score":           csat,
            "nps_flag":             nps_flag,
        })

# ── Build the DataFrame and save ─────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDONE: qsr_customer_survey.csv saved — {len(df):,} rows | {len(df.columns)} columns")
print()

# ── Verification summary ──────────────────────────────────────────────────────
# Like reading your weekly guest satisfaction report — a quick dashboard
# of how the data looks before it goes into the platform.
print("=" * 65)
print("  CUSTOMER SURVEY SUMMARY — 2024")
print("=" * 65)

print(f"\n  Total responses   : {len(df):,}")
print(f"  Date range        : {df['date'].min()}  to  {df['date'].max()}")
print(f"  Avg per day       : {len(df) / df['date'].nunique():.1f}")

print(f"\n  AVERAGE RATINGS (out of 5.0):")
rating_cols = [
    "speed_of_service", "staff_friendliness", "order_accuracy",
    "cleanliness", "product_availability", "overall_satisfaction"
]
for col in rating_cols:
    avg = df[col].mean()
    bar = "|" * int(avg * 4)   # simple visual bar (ASCII-safe)
    print(f"    {col:<24} {avg:.2f}  {bar}")

print(f"\n  CSAT SCORE (avg of 5 ratings): {df['csat_score'].mean():.2f} / 5.00")

print(f"\n  NPS BREAKDOWN:")
nps_counts = df["nps_flag"].value_counts()
total = len(df)
for flag in ["Promoter", "Passive", "Detractor"]:
    count = nps_counts.get(flag, 0)
    pct   = count / total * 100
    print(f"    {flag:<12} {count:>5} ({pct:.1f}%)")

promoters   = nps_counts.get("Promoter",  0) / total * 100
detractors  = nps_counts.get("Detractor", 0) / total * 100
nps_score   = round(promoters - detractors, 1)
print(f"    NPS Score = Promoters% - Detractors% = {nps_score:+.1f}")

print(f"\n  SHIFT PERIOD BREAKDOWN:")
shift_counts = df["shift_period"].value_counts()
for period in ["morning", "afternoon", "evening"]:
    count = shift_counts.get(period, 0)
    avg_overall = df[df["shift_period"] == period]["overall_satisfaction"].mean()
    print(f"    {period:<12} {count:>5} responses  |  avg satisfaction {avg_overall:.2f}")

print(f"\n  WORST-SCORING CONDITIONS (avg overall satisfaction):")
# Weekend vs weekday
wknd_avg = df[df["is_weekend"] == 1]["overall_satisfaction"].mean()
wkdy_avg = df[df["is_weekend"] == 0]["overall_satisfaction"].mean()
print(f"    Weekend avg : {wknd_avg:.2f}  |  Weekday avg : {wkdy_avg:.2f}")

# Holiday vs normal
hol_avg  = df[df["date"].isin(
    sales[sales["is_holiday"] == 1]["date"].unique()
)]["overall_satisfaction"].mean()
norm_avg = df[~df["date"].isin(
    sales[sales["is_holiday"] == 1]["date"].unique()
)]["overall_satisfaction"].mean()
print(f"    Holiday avg : {hol_avg:.2f}  |  Normal day  : {norm_avg:.2f}")
