"""
Stage 1e — Generate Employee Morale Survey Data
================================================
WHY THIS MATTERS (Business Reason):
  This is the most important dataset in the entire platform.

  The central business theory is:
    Better scheduling -> higher staff morale -> better customer experience
    -> consistent sales growth.

  The customer survey (Stage 1d) gives us the RIGHT side of that equation.
  This dataset gives us the LEFT side — what employees actually feel each week
  about the quality of their scheduling and work environment.

  Each of the 6 survey questions maps directly to one of the 6 Morale Factors
  that this platform is designed to protect:

    Factor 1 -> hours_consistency       (were my hours predictable this week?)
    Factor 2 -> comms_training          (did I get clear info and training?)
    Factor 3 -> manager_floor_presence  (was a manager visible during peaks?)
    Factor 4 -> product_availability    (did we run out of anything?)
    Factor 5 -> personal_time_respect   (were my days off and schedule honored?)
    Factor 6 -> daily_plan_clarity      (did I know the goals at shift start?)

  The morale_index (weighted average of all 6) feeds Module 3 Hiring Trigger T4:
    "Employee morale_index < 3.0 for 3+ consecutive weeks → flag for intervention"

  When you can prove that scheduling decisions (Module 1) move the morale_index,
  you have the data spine that connects all three modules.

WHAT THIS SCRIPT DOES:
  1. Loads the employee roster to get IDs, names, and roles
  2. Generates one survey response per employee per week (52 weeks × 15 = 780)
  3. Applies realistic role-based morale baselines:
       - Managers score higher — they control the environment
       - Supervisors are solid but feel the middle-management squeeze
       - Full-time crew are schedule-sensitive — inconsistency hits them hard
       - Part-time crew are the most vulnerable — outside commitments collide
  4. Simulates two "rough patch" periods for real employees to trigger T4:
       - E012 Emma Leblanc  → weeks 20-26 (mid-year scheduling disruption)
       - E015 Marco Ferreira → weeks 35-42 (fall burnout stretch)
  5. Calculates morale_index, tracks consecutive low weeks, sets at_risk_flag
  6. Saves to data/qsr_employee_survey.csv

THE 17 COLUMNS:
  survey_id            — unique ID (ES0001, ES0002, ...)
  week_number          — ISO week (1-52)
  week_start_date      — Monday date of that week (YYYY-MM-DD)
  employee_id          — matches emp_id in qsr_employees.csv
  role                 — employee's current role
  hours_scheduled      — actual hours assigned that week
  hours_consistency    — rating 1-5 (Factor 1: were hours predictable?)
  comms_training       — rating 1-5 (Factor 2: clear communication & training?)
  manager_floor_presence — rating 1-5 (Factor 3: manager visible at peaks?)
  product_availability — rating 1-5 (Factor 4: no stock runouts?)
  personal_time_respect — rating 1-5 (Factor 5: days off honored?)
  daily_plan_clarity   — rating 1-5 (Factor 6: shift goals were clear?)
  overall_morale       — rating 1-5 (direct question: how's your morale?)
  retention_intent     — rating 1-5 (5 = definitely staying, 1 = looking to leave)
  morale_index         — float, weighted avg of the 6 factor scores
  consecutive_low_weeks — how many weeks in a row morale_index was below 3.0
  at_risk_flag         — 1 if consecutive_low_weeks >= 3, else 0

Outputs:
  data/qsr_employee_survey.csv   (780 rows, 17 columns)
"""

import pandas as pd    # our spreadsheet tool
import numpy as np     # math and random numbers
import os              # file path builder

# ── Random seed ──────────────────────────────────────────────────────────────
# Same reason as Stage 1d — fixed seed means reproducible data every run.
# Think of it as "shuffling the deck the same way every time."
np.random.seed(7)

# ── File paths ────────────────────────────────────────────────────────────────
BASE        = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EMP_PATH    = os.path.join(BASE, "data", "qsr_employees.csv")
SALES_PATH  = os.path.join(BASE, "data", "qsr_sales_data.csv")
OUTPUT_PATH = os.path.join(BASE, "data", "qsr_employee_survey.csv")

# ── Load employees ────────────────────────────────────────────────────────────
print("Loading employee roster...")
employees = pd.read_csv(EMP_PATH)
print(f"  {len(employees)} employees loaded")

# ── Build week lookup: week_number -> week_start_date ─────────────────────────
# We pull the Monday date for each ISO week directly from the sales data.
# This keeps week numbering perfectly consistent across all datasets.
print("Building 2024 week calendar...")
sales = pd.read_csv(SALES_PATH)
sales = sales[sales["year"] == 2024].copy()

week_lookup = (
    sales[["iso_week", "date"]]
    .drop_duplicates("iso_week")
    .sort_values("iso_week")
    .rename(columns={"iso_week": "week_number"})
)
# Convert to Monday date — pandas gives us the first occurrence per week,
# which from our data is always a Monday.
week_lookup["week_start_date"] = pd.to_datetime(week_lookup["date"]).apply(
    lambda d: (d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d")
)
week_lookup = week_lookup[["week_number", "week_start_date"]]
print(f"  {len(week_lookup)} weeks found (ISO weeks 1-52 of 2024)")

# ── Role-based morale profiles ────────────────────────────────────────────────
# In any restaurant, morale follows a predictable hierarchy.
# Managers set the tone — they score higher because they *are* the environment.
# Supervisors are solid but feel pressure from both sides.
# Full-time crew are deeply affected by schedule consistency.
# Part-timers are most sensitive because their outside life competes hardest.
#
# Each profile is a dict of base scores for the 6 morale factors.
# Think of it as the "typical week" for that type of employee.

ROLE_PROFILES = {
    "Manager": {
        "hours_consistency":      4.4,   # managers set their own hours
        "comms_training":         4.5,   # they lead the comms, so they're informed
        "manager_floor_presence": 4.6,   # they ARE the manager on floor
        "product_availability":   4.3,   # they control ordering
        "personal_time_respect":  4.2,   # mostly honored but on-call culture
        "daily_plan_clarity":     4.6,   # they write the daily plan
        "base_morale":            4.3,   # overall direct morale rating
        "base_retention":         4.5,   # managers tend to stay longer
        "hours_range":            (38, 44),  # weekly hours
        "noise_std":              0.30,  # low variation — stable role
    },
    "Shift Supervisor": {
        "hours_consistency":      4.0,
        "comms_training":         4.1,
        "manager_floor_presence": 4.2,
        "product_availability":   4.0,
        "personal_time_respect":  3.9,
        "daily_plan_clarity":     4.1,
        "base_morale":            3.9,
        "base_retention":         4.1,
        "hours_range":            (32, 40),
        "noise_std":              0.35,
    },
    "Crew - Full Time": {
        "hours_consistency":      3.7,   # schedule changes affect them most
        "comms_training":         3.7,
        "manager_floor_presence": 3.8,
        "product_availability":   3.8,
        "personal_time_respect":  3.7,
        "daily_plan_clarity":     3.7,
        "base_morale":            3.6,
        "base_retention":         3.8,
        "hours_range":            (28, 38),
        "noise_std":              0.45,  # more variation week to week
    },
    "Crew - Part Time": {
        "hours_consistency":      3.4,   # PT schedules are most unpredictable
        "comms_training":         3.5,
        "manager_floor_presence": 3.5,
        "product_availability":   3.6,
        "personal_time_respect":  3.4,   # days off are the most sensitive factor
        "daily_plan_clarity":     3.5,
        "base_morale":            3.3,
        "base_retention":         3.5,
        "hours_range":            (12, 22),
        "noise_std":              0.55,  # most variable — outside life intrudes
    },
}

# ── Rough patch definitions ───────────────────────────────────────────────────
# Two employees will go through a stretch of low morale — enough to trigger T4.
# This is realistic: in any store, 1-2 employees hit a rough patch each year.
# The at_risk_flag catches them before they quit.
#
# ROUGH_PATCHES format:
#   emp_id -> { weeks: range of weeks, penalties: { factor: amount_to_subtract } }

ROUGH_PATCHES = {
    "E012": {   # Emma Leblanc — mid-year scheduling disruption (weeks 20-26)
        # Emma is PT midday (Mon off, Fri off). Imagine her school schedule
        # changed and the store wasn't adjusting her shifts quickly.
        "weeks": range(20, 27),    # 7 rough weeks
        "penalties": {
            "hours_consistency":    -1.6,   # hours were all over the place
            "personal_time_respect":-1.5,   # days off being overridden
            "daily_plan_clarity":   -0.8,   # no one briefed her properly
            "base_morale":          -1.8,   # direct morale tanks
            "base_retention":       -1.5,   # starting to think about leaving
        },
    },
    "E015": {   # Marco Ferreira — fall burnout stretch (weeks 35-42)
        # Marco is PT evening (Sun off, Mon off). Fall is his busiest semester.
        # He's being scheduled too close to his availability limits.
        "weeks": range(35, 43),    # 8 rough weeks
        "penalties": {
            "hours_consistency":    -1.4,
            "personal_time_respect":-1.8,   # schedule not respecting his life
            "comms_training":       -0.9,   # feeling left out of the loop
            "base_morale":          -1.9,
            "base_retention":       -1.7,
        },
    },
}

# ── Morale index weights ──────────────────────────────────────────────────────
# These weights reflect how much each factor contributes to overall morale.
# Hours consistency and personal time respect carry the most weight —
# from 20 years of QSR experience, these are the #1 and #2 reasons staff quit.
MORALE_WEIGHTS = {
    "hours_consistency":      0.20,
    "comms_training":         0.15,
    "manager_floor_presence": 0.15,
    "product_availability":   0.15,
    "personal_time_respect":  0.20,
    "daily_plan_clarity":     0.15,
}

# ── Helper: clip a score to valid range 1.0–5.0 ──────────────────────────────
def clip_score(value: float) -> float:
    # Like enforcing a "no score below 1, no score above 5" rule on a paper form
    return float(np.clip(round(value, 1), 1.0, 5.0))

# ── Generate all survey rows ──────────────────────────────────────────────────
print("Generating employee survey responses (15 employees x 52 weeks)...")

rows           = []
survey_counter = 0

# Loop through each employee
for _, emp in employees.iterrows():

    role        = emp["role"]
    emp_id      = emp["emp_id"]
    profile     = ROLE_PROFILES[role]    # get this employee's morale profile
    rough_patch = ROUGH_PATCHES.get(emp_id, None)  # any rough period planned?

    # Track consecutive weeks below 3.0 — resets when morale recovers
    # Think of it like a streak counter: "3rd week in a row struggling"
    consecutive_low = 0

    # Loop through all 52 weeks
    for _, week_row in week_lookup.iterrows():

        week_num        = int(week_row["week_number"])
        week_start_date = week_row["week_start_date"]

        # ── Apply rough patch penalties if this week falls in the bad stretch ──
        # Like a manager who went on leave for two months — you'd expect morale
        # to dip while the team feels unsupported.
        in_rough_patch = (rough_patch is not None) and (week_num in rough_patch["weeks"])
        penalties      = rough_patch["penalties"] if in_rough_patch else {}

        # ── Generate each of the 6 factor scores ─────────────────────────────
        factor_scores = {}
        for factor in MORALE_WEIGHTS:
            base     = profile[factor]                      # role baseline
            penalty  = penalties.get(factor, 0.0)          # rough patch hit
            noise    = np.random.normal(0, profile["noise_std"])  # weekly variation
            score    = clip_score(base + penalty + noise)
            factor_scores[factor] = score

        # ── Calculate morale_index ────────────────────────────────────────────
        # Weighted average of the 6 factor scores.
        # This is the single number we watch for Trigger T4.
        morale_index = round(
            sum(factor_scores[f] * MORALE_WEIGHTS[f] for f in MORALE_WEIGHTS), 2
        )

        # ── Track consecutive low weeks ───────────────────────────────────────
        # Like watching whether an employee is late 3 Mondays in a row —
        # one late is noise, three in a row is a pattern that needs attention.
        if morale_index < 3.0:
            consecutive_low += 1
        else:
            consecutive_low = 0   # streak broken — reset the counter

        # at_risk_flag fires when 3 or more consecutive weeks below 3.0
        at_risk_flag = 1 if consecutive_low >= 3 else 0

        # ── Overall morale (direct self-rating) ───────────────────────────────
        # Separate from the index — this is the employee's gut-feel answer
        # to "how is your morale this week?" Not always rational, and often
        # slightly more extreme than the calculated index.
        morale_noise  = np.random.normal(0, 0.35)
        base_morale   = profile["base_morale"] + penalties.get("base_morale", 0.0)
        overall_morale = clip_score(base_morale + morale_noise)

        # ── Retention intent ──────────────────────────────────────────────────
        # "On a scale of 1-5, how likely are you to still be here in 6 months?"
        # Closely tied to morale, but also influenced by role stability.
        ret_noise = np.random.normal(0, 0.40)
        base_ret  = profile["base_retention"] + penalties.get("base_retention", 0.0)
        retention_intent = clip_score(base_ret + ret_noise)

        # ── Hours scheduled ───────────────────────────────────────────────────
        # Generated within the role's typical range with some weekly variation.
        # During rough patches for PT staff, hours become more erratic.
        low_h, high_h = profile["hours_range"]
        if in_rough_patch and role == "Crew - Part Time":
            # Hours swing wider during the rough patch — that's part of the problem
            hours = int(np.clip(np.random.normal((low_h + high_h) / 2, 5), low_h - 4, high_h + 4))
        else:
            hours = int(np.clip(np.random.normal((low_h + high_h) / 2, 3), low_h, high_h))

        # ── Assemble the row ──────────────────────────────────────────────────
        survey_counter += 1
        rows.append({
            "survey_id":              f"ES{survey_counter:04d}",
            "week_number":            week_num,
            "week_start_date":        week_start_date,
            "employee_id":            emp_id,
            "role":                   role,
            "hours_scheduled":        hours,
            "hours_consistency":      factor_scores["hours_consistency"],
            "comms_training":         factor_scores["comms_training"],
            "manager_floor_presence": factor_scores["manager_floor_presence"],
            "product_availability":   factor_scores["product_availability"],
            "personal_time_respect":  factor_scores["personal_time_respect"],
            "daily_plan_clarity":     factor_scores["daily_plan_clarity"],
            "overall_morale":         overall_morale,
            "retention_intent":       retention_intent,
            "morale_index":           morale_index,
            "consecutive_low_weeks":  consecutive_low,
            "at_risk_flag":           at_risk_flag,
        })

# ── Build DataFrame and save ─────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDONE: qsr_employee_survey.csv saved — {len(df):,} rows | {len(df.columns)} columns")
print()

# ── Verification summary ──────────────────────────────────────────────────────
print("=" * 65)
print("  EMPLOYEE SURVEY SUMMARY — 2024 (52 weeks)")
print("=" * 65)

print(f"\n  Total responses   : {len(df):,}")
print(f"  Employees covered : {df['employee_id'].nunique()}")
print(f"  Weeks covered     : {df['week_number'].nunique()}")

# ── Average morale index by role ──────────────────────────────────────────────
print(f"\n  AVERAGE MORALE INDEX BY ROLE:")
role_morale = (
    df.groupby("role")["morale_index"]
    .mean()
    .sort_values(ascending=False)
)
for role, avg in role_morale.items():
    bar = "|" * int(avg * 4)
    print(f"    {role:<22} {avg:.2f}  {bar}")

# ── Average morale index per employee ────────────────────────────────────────
print(f"\n  AVERAGE MORALE INDEX PER EMPLOYEE:")
emp_morale = (
    df.groupby(["employee_id", "role"])["morale_index"]
    .mean()
    .reset_index()
    .sort_values("morale_index", ascending=False)
)
# Also get the employee name from the employee file
emp_names = employees[["emp_id", "name"]].rename(columns={"emp_id": "employee_id"})
emp_morale = emp_morale.merge(emp_names, on="employee_id")
for _, row in emp_morale.iterrows():
    flag = " <-- LOW AVG" if row["morale_index"] < 3.3 else ""
    print(f"    {row['employee_id']}  {row['name']:<22} {row['morale_index']:.2f}{flag}")

# ── At-risk employees (T4 trigger) ────────────────────────────────────────────
print(f"\n  TRIGGER T4 — AT-RISK EMPLOYEES (morale_index < 3.0 for 3+ consecutive weeks):")
at_risk = df[df["at_risk_flag"] == 1]
if at_risk.empty:
    print("    None triggered — all employees above threshold.")
else:
    at_risk_summary = (
        at_risk.groupby(["employee_id"])
        .agg(
            at_risk_weeks  = ("at_risk_flag", "sum"),
            worst_index    = ("morale_index", "min"),
            weeks_affected = ("week_number", lambda x: f"wk {x.min()}-{x.max()}")
        )
        .reset_index()
    )
    at_risk_summary = at_risk_summary.merge(emp_names, on="employee_id")
    for _, row in at_risk_summary.iterrows():
        print(f"    {row['employee_id']}  {row['name']:<22}  "
              f"{row['at_risk_weeks']} at-risk weeks  |  "
              f"worst index: {row['worst_index']:.2f}  |  "
              f"{row['weeks_affected']}")

# ── Average 6 factor scores overall ─────────────────────────────────────────
print(f"\n  AVERAGE FACTOR SCORES (all employees, full year):")
factor_cols = [
    "hours_consistency", "comms_training", "manager_floor_presence",
    "product_availability", "personal_time_respect", "daily_plan_clarity"
]
for col in factor_cols:
    avg = df[col].mean()
    bar = "|" * int(avg * 4)
    print(f"    {col:<26} {avg:.2f}  {bar}")

# ── Retention risk flag ───────────────────────────────────────────────────────
print(f"\n  RETENTION RISK (avg retention_intent < 3.0):")
low_retention = (
    df.groupby("employee_id")["retention_intent"]
    .mean()
    .reset_index()
)
low_retention = low_retention[low_retention["retention_intent"] < 3.3]
low_retention = low_retention.merge(emp_names, on="employee_id")
if low_retention.empty:
    print("    No employees in critical retention risk range.")
else:
    for _, row in low_retention.iterrows():
        print(f"    {row['employee_id']}  {row['name']:<22}  avg retention intent: {row['retention_intent']:.2f}")
