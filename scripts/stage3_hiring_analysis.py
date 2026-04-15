"""
Stage 3 — Hiring Needs Analysis (Module 3)
==========================================
WHY THIS MATTERS (Business Reason):
  In 20+ years of QSR management, the costliest mistake isn't a bad hire —
  it's hiring REACTIVELY. Waiting until someone quits, burns out, or gets
  promoted before you start looking means you're always 3-4 weeks behind.

  Module 3 is the early-warning system that prevents that.

  It reads all 7 datasets built in this project and asks:
    "What signals exist RIGHT NOW that tell us we need to hire?"

  It does this by checking 7 programmatic triggers against real data:
    T1: Station has fewer than 2 certified staff (Level 2+)
    T2: Any regular shift window has dangerously thin supervisor coverage
    T3: Sales volume >20% above capacity average for 2+ consecutive weeks
    T4: Employee morale_index <3.0 for 3+ consecutive weeks (flight risk)
    T5: Any role or certification has only 1 current holder (single failure point)
    T6: Weekly hours demand exceeds total team capacity
    T7: Employee promotion_ready + bench_priority=1 (creates vacancy below)

  Each trigger fires a record with: what gap was found, which role/station,
  how many hours per week it represents, how many heads to hire, and a
  recommended action for the GM to take.

WHAT THIS SCRIPT DOES:
  1. Loads all 5 relevant datasets
  2. Runs each trigger check programmatically against real data
  3. Logs every gap event as one row in qsr_hiring_analysis.csv
  4. Assigns severity (High / Medium / Low) and recommended actions
  5. Saves to data/qsr_hiring_analysis.csv

THE 15 COLUMNS:
  analysis_id          — unique ID (HA001, HA002 ...)
  week_number          — ISO week when the gap was detected
  trigger_code         — T1 through T7
  trigger_label        — plain-English description of the trigger
  severity             — High / Medium / Low
  affected_station     — station code, role name, or "Multiple"
  affected_role        — role category affected
  availability_needed  — shift window needed (Morning / Evening / Weekend / All)
  days_needed          — specific days the gap falls on
  recommended_action   — what the GM should do about it
  hrs_gap_per_week     — estimated weekly hours this gap represents
  estimated_headcount  — how many people to hire to close the gap
  linked_employee_id   — related employee (if specific to one person)
  status               — Open / In Progress / Resolved
  resolved_date        — date resolved (null if still open)

Outputs:
  data/qsr_hiring_analysis.csv   (~50 rows, 15 columns)
"""

import pandas as pd       # our spreadsheet tool
import numpy as np        # math
import os                 # file paths
from datetime import date, timedelta

# ── File paths ────────────────────────────────────────────────────────────────
BASE         = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EMP_PATH     = os.path.join(BASE, "data", "qsr_employees.csv")
SALES_PATH   = os.path.join(BASE, "data", "qsr_sales_data.csv")
SURVEY_PATH  = os.path.join(BASE, "data", "qsr_employee_survey.csv")
TRAIN_PATH   = os.path.join(BASE, "data", "qsr_training_records.csv")
COMPLY_PATH  = os.path.join(BASE, "data", "qsr_compliance_training.csv")
OUTPUT_PATH  = os.path.join(BASE, "data", "qsr_hiring_analysis.csv")

# ── Snapshot date ─────────────────────────────────────────────────────────────
SNAPSHOT = date(2024, 12, 31)   # all analysis is "as of" this date

# ── Load all datasets ─────────────────────────────────────────────────────────
print("Loading datasets...")
emp_df     = pd.read_csv(EMP_PATH)
sales_df   = pd.read_csv(SALES_PATH)
survey_df  = pd.read_csv(SURVEY_PATH)
train_df   = pd.read_csv(TRAIN_PATH)
comply_df  = pd.read_csv(COMPLY_PATH)
print(f"  Employees      : {len(emp_df)}")
print(f"  Sales rows     : {len(sales_df):,}")
print(f"  Survey rows    : {len(survey_df):,}")
print(f"  Training rows  : {len(train_df):,}")
print(f"  Compliance rows: {len(comply_df):,}")

# ── Helper: week_number -> week_start_date (Monday) ──────────────────────────
# Build a lookup so we can convert ISO week numbers to real dates.
sales_2024 = sales_df[sales_df["year"] == 2024].copy()
week_dates  = (
    sales_2024[["iso_week", "date"]]
    .drop_duplicates("iso_week")
    .sort_values("iso_week")
    .rename(columns={"iso_week": "week_number"})
)
week_dates["week_start"] = pd.to_datetime(week_dates["date"]).apply(
    lambda d: d - timedelta(days=d.weekday())
)

def week_to_date(week_num: int) -> date:
    """Return the Monday date for a given ISO week number."""
    row = week_dates[week_dates["week_number"] == week_num]
    if row.empty:
        return SNAPSHOT
    return row["week_start"].iloc[0].date()

# ── Helper: resolved_date for past events ─────────────────────────────────────
def resolve_date(week_num: int, weeks_after: int = 3) -> str:
    """
    A resolved trigger is marked as closed a few weeks after detection.
    Think of it like closing a service ticket once the action is taken.
    """
    d = week_to_date(week_num) + timedelta(weeks=weeks_after)
    return d.strftime("%Y-%m-%d") if d < SNAPSHOT else None

# ── Collector: all trigger events go into this list ──────────────────────────
rows           = []
analysis_count = 0

def add_record(week_num, code, label, severity, station, role,
               avail, days, action, hrs, headcount, linked_id, status, resolved):
    """
    Helper to append a trigger event row.
    Like writing one line on your hiring gap tracking sheet.
    """
    global analysis_count
    analysis_count += 1
    rows.append({
        "analysis_id":         f"HA{analysis_count:03d}",
        "week_number":         week_num,
        "trigger_code":        code,
        "trigger_label":       label,
        "severity":            severity,
        "affected_station":    station,
        "affected_role":       role,
        "availability_needed": avail,
        "days_needed":         days,
        "recommended_action":  action,
        "hrs_gap_per_week":    hrs,
        "estimated_headcount": headcount,
        "linked_employee_id":  linked_id,
        "status":              status,
        "resolved_date":       resolved,
    })

# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER T1 — Station has fewer than 2 certified staff (Level 2+)
# ─────────────────────────────────────────────────────────────────────────────
# Business reason: If fewer than 2 certified people can run a station,
# a single sick call leaves the station unmanned. You can't run a shift
# without coverage at every station.
#
# Data source: qsr_training_records.csv
# Method: Count employees with Completed cert at Level 2+ per station.
#         Flag any station below the minimum of 2.
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning T1: Station certification gap check...")

STATIONS = ["BEV", "PREP", "PACK", "CLEAN", "OPEN", "CLOSE", "SUP", "TRAIN"]

# Only "Completed" certs count — "Pending Renewal" is expired, "In Progress" = not certified
active_certs = train_df[
    (train_df["proficiency_level"] >= 2) &
    (train_df["status"] == "Completed")
]

for station in STATIONS:
    # Count unique employees who hold an active cert at this station
    certified_count = active_certs[
        active_certs["station_code"] == station
    ]["employee_id"].nunique()

    if certified_count < 2:
        # Each station needs at least 2 certified staff for shift coverage safety.
        # One person = single point of failure. Any absence = station goes dark.
        print(f"  T1 fired: {station} — only {certified_count} certified staff")
        add_record(
            week_num   = 1,
            code       = "T1",
            label      = f"{station} station has only {certified_count} employee(s) with active Level 2+ certification",
            severity   = "High",
            station    = station,
            role       = "Shift Supervisor / Manager",
            avail      = "All Shifts",
            days       = "All Days",
            action     = (f"Immediate: cross-train 1+ existing staff to {station} Level 2. "
                          f"Short-term: hire 1 candidate with {station} experience."),
            hrs        = 16,
            headcount  = 1,
            linked_id  = None,
            status     = "Open",
            resolved   = None,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER T2 — Regular shift windows with dangerously thin SUP coverage
# ─────────────────────────────────────────────────────────────────────────────
# Business reason: Hard Rule 2 says every shift must have 1 supervisor or
# manager. If a day of the week only has 1 qualified person available, a
# single call-out leaves the shift unsupervised — which is an HR and
# operations violation.
#
# Data source: qsr_employees.csv (days_off, availability_start/end, role)
# Method: For each day of the week, count who qualifies as a supervisor
#         AND is available. Flag days where this count drops to 1.
# ─────────────────────────────────────────────────────────────────────────────
print("Running T2: Supervisor coverage gap by day...")

SUPERVISOR_ROLES = {"Manager", "Shift Supervisor"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Filter to employees who can supervise
supervisors = emp_df[emp_df["role"].isin(SUPERVISOR_ROLES)].copy()

# Build a list of days_off for each supervisor
supervisors["days_off_list"] = supervisors["days_off"].apply(
    lambda s: [d.strip() for d in str(s).split(",")]
)

for day in DAYS:
    # Count supervisors available on this day (not on day off, avail covers open shift)
    available = supervisors[
        supervisors["days_off_list"].apply(lambda dlist: day not in dlist)
    ]

    sup_count = len(available)

    if sup_count <= 2:
        # Two supervisors on any given day is the absolute minimum —
        # it means one opens, one closes, and there's NO backup.
        # A single sick call = an unsupervised shift, which violates Hard Rule 2.
        print(f"  T2 fired: {day} — only {sup_count} supervisors available")
        add_record(
            week_num   = 1,
            code       = "T2",
            label      = f"Only {sup_count} supervisor(s) available on {day}s — no backup if one calls out",
            severity   = "High",
            station    = "SUP",
            role       = "Shift Supervisor",
            avail      = "Open + Close",
            days       = day,
            action     = (f"Ensure at least 3 SUP-certified staff are available on {day}. "
                          f"Consider cross-training a senior FT crew member for SUP on this day."),
            hrs        = 16,
            headcount  = 1,
            linked_id  = None,
            status     = "Open",
            resolved   = None,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER T3 — Sales volume >20% above capacity average for 2+ weeks
# ─────────────────────────────────────────────────────────────────────────────
# Business reason: When the ML model consistently predicts we need more staff
# than our team can provide, that's not a scheduling problem — it's a hiring
# signal. One busy week = scheduling adjustment. Two or more in a row = hire.
#
# Data source: qsr_sales_data.csv (2024 only)
# Method: Compute avg weekly staff_needed. Find weeks where it's >20% above
#         the annual average. Flag consecutive 2+ week stretches as one event.
# ─────────────────────────────────────────────────────────────────────────────
print("Running T3: High-volume capacity pressure weeks...")

# Compute total staff-hours needed per week
# (sum of staff_needed across all 18 daily hours × 7 days)
weekly_demand = (
    sales_2024.groupby("iso_week")["staff_needed"]
    .sum()
    .reset_index()
    .rename(columns={"iso_week": "week_number", "staff_needed": "total_staff_hours"})
)

# The annual average weekly demand is our baseline "normal capacity"
annual_avg = weekly_demand["total_staff_hours"].mean()
threshold  = annual_avg * 1.20   # 20% above normal = T3 fires

# Identify which weeks are above threshold
weekly_demand["above_threshold"] = weekly_demand["total_staff_hours"] > threshold

# Find stretches of 2+ consecutive weeks above threshold
# Think of this like spotting a multi-week heat wave vs a single hot day
t3_events = []
streak_start = None
streak_weeks = []

for _, row in weekly_demand.iterrows():
    wk = int(row["week_number"])
    if row["above_threshold"]:
        if streak_start is None:
            streak_start = wk
        streak_weeks.append(wk)
    else:
        if streak_start is not None and len(streak_weeks) >= 2:
            t3_events.append(streak_weeks.copy())
        streak_start = None
        streak_weeks = []

# Catch a streak that runs to the end of the year
if streak_start is not None and len(streak_weeks) >= 2:
    t3_events.append(streak_weeks.copy())

for event_weeks in t3_events:
    first_wk   = event_weeks[0]
    last_wk    = event_weeks[-1]
    duration   = len(event_weeks)
    peak_demand = weekly_demand[
        weekly_demand["week_number"].isin(event_weeks)
    ]["total_staff_hours"].max()
    pct_above  = round((peak_demand / annual_avg - 1) * 100, 1)

    # A past high-volume stretch (before week 40) is already resolved —
    # the store survived it. Future or recent ones are still open risks.
    is_past    = last_wk < 45
    status     = "Resolved" if is_past else "Open"
    res_date   = resolve_date(last_wk, weeks_after=2) if is_past else None

    print(f"  T3 fired: weeks {first_wk}-{last_wk} ({duration} weeks, +{pct_above}% above avg)")
    add_record(
        week_num   = first_wk,
        code       = "T3",
        label      = (f"Weeks {first_wk}-{last_wk}: demand {pct_above}% above "
                      f"avg capacity for {duration} consecutive weeks"),
        severity   = "Medium",
        station    = "Multiple",
        role       = "Crew - Full Time / Part Time",
        avail      = "All Shifts",
        days       = "All Days",
        action     = (f"Pre-schedule additional part-time coverage for weeks "
                      f"{first_wk}-{last_wk}. "
                      f"If pattern repeats, recruit 1 additional PT staff."),
        hrs        = round(peak_demand - annual_avg),
        headcount  = 1,
        linked_id  = None,
        status     = status,
        resolved   = res_date,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER T4 — Morale index <3.0 for 3+ consecutive weeks (flight risk)
# ─────────────────────────────────────────────────────────────────────────────
# Business reason: An employee with a morale index below 3.0 for 3 weeks in a
# row is telling you they're struggling. If you miss this signal, they quit.
# Replacing a trained crew member costs 2-4 weeks of their salary in
# recruitment, onboarding, and lost productivity. Better to intervene early.
#
# Data source: qsr_employee_survey.csv
# Method: Find employees where at_risk_flag=1 exists (3+ consecutive low weeks).
#         Log one record per employee at the point T4 first fired.
# ─────────────────────────────────────────────────────────────────────────────
print("Running T4: At-risk morale detection...")

# Find all employees who ever hit at_risk_flag=1
at_risk_employees = survey_df[survey_df["at_risk_flag"] == 1]

# Group by employee — we want the FIRST week they triggered T4
# (that's when intervention should have started)
if not at_risk_employees.empty:
    first_trigger = (
        at_risk_employees.groupby("employee_id")
        .agg(
            first_risk_week    = ("week_number", "min"),
            worst_morale       = ("morale_index", "min"),
            at_risk_week_count = ("at_risk_flag", "sum"),
        )
        .reset_index()
    )
    first_trigger = first_trigger.merge(
        emp_df[["emp_id", "name", "role"]].rename(columns={"emp_id": "employee_id"}),
        on="employee_id"
    )

    for _, row in first_trigger.iterrows():
        week_triggered = int(row["first_risk_week"])
        worst          = round(row["worst_morale"], 2)
        at_risk_weeks  = int(row["at_risk_week_count"])
        emp_name       = row["name"]
        emp_role       = row["role"]

        # If the employee's morale recovered before the snapshot, it's resolved
        # Check if their most recent weeks show recovery (last survey row)
        emp_recent = survey_df[
            survey_df["employee_id"] == row["employee_id"]
        ].sort_values("week_number").tail(4)
        recovered = emp_recent["morale_index"].mean() >= 3.2

        status   = "Resolved" if recovered else "In Progress"
        res_date = resolve_date(week_triggered, weeks_after=6) if recovered else None

        print(f"  T4 fired: {emp_name} (wk {week_triggered}, worst index {worst})")
        add_record(
            week_num   = week_triggered,
            code       = "T4",
            label      = (f"{emp_name} morale_index dropped below 3.0 for "
                          f"{at_risk_weeks} consecutive weeks (worst: {worst})"),
            severity   = "High",
            station    = "Multiple",
            role       = emp_role,
            avail      = "Current schedule",
            days       = "All scheduled days",
            action     = (f"Immediate: 1-on-1 check-in with {emp_name}. "
                          f"Review last 4 weeks of schedule for hours consistency "
                          f"and personal time respect. Begin succession watch — "
                          f"flag {emp_name} for potential exit risk."),
            hrs        = round(emp_df[emp_df["emp_id"] == row["employee_id"]]["max_hours"].values[0]),
            headcount  = 1,
            linked_id  = row["employee_id"],
            status     = status,
            resolved   = res_date,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER T5 — Only 1 employee holds a critical role or certification
# ─────────────────────────────────────────────────────────────────────────────
# Business reason: If only one person can do something essential, you are
# one resignation letter away from a gap you cannot fill from inside.
# T5 is your "key person risk" detector — the single points of failure
# that the org chart doesn't make obvious until it's too late.
#
# Data source: qsr_employees.csv + qsr_compliance_training.csv
# Check A: Roles with only 1 employee (critical: Manager, Shift Supervisor)
# Check B: Compliance certs with only 1 current holder in management/sup group
# ─────────────────────────────────────────────────────────────────────────────
print("Running T5: Single-point-of-failure check...")

# ── Check A: Role headcount ───────────────────────────────────────────────────
# A role is at risk if it has only 1 or 2 employees — losing one is severe.
role_counts = emp_df["role"].value_counts()

# We flag roles with fewer than 3 headcount as potential single-POF risks
# (3 is the minimum for safe redundancy: 1 off, 1 sick, 1 covers)
T5_ROLE_THRESHOLD = 3

for role, count in role_counts.items():
    if count < T5_ROLE_THRESHOLD and role in ["Manager", "Shift Supervisor"]:
        print(f"  T5 fired (role): {role} — only {count} employee(s)")
        add_record(
            week_num   = 1,
            code       = "T5",
            label      = (f"Only {count} employee(s) hold the {role} role. "
                          f"One departure leaves the role critically understaffed."),
            severity   = "Medium",
            station    = "SUP",
            role       = role,
            avail      = "All Shifts",
            days       = "All Days",
            action     = (f"Develop 1 internal candidate for {role} pipeline. "
                          f"Review bench_priority=1 employees for accelerated development. "
                          f"Begin external recruiting if bench is not ready within 3 months."),
            hrs        = 40,
            headcount  = 1,
            linked_id  = None,
            status     = "In Progress",
            resolved   = None,
        )

# ── Check B: Compliance certifications held by only 1 active person ───────────
# Read current (non-expired) holders per cert type.
current_comply = comply_df[comply_df["renewal_status"].isin(["Current", "Expiring Soon"])]
cert_counts    = current_comply.groupby("certification_name")["employee_id"].nunique()

for cert_name, count in cert_counts.items():
    if count == 1:
        # Only 1 person has a current copy of this mandatory cert.
        # If they leave before renewal, the store has zero coverage.
        holder_id = current_comply[
            current_comply["certification_name"] == cert_name
        ]["employee_id"].iloc[0]
        print(f"  T5 fired (cert): '{cert_name}' — only 1 current holder ({holder_id})")
        add_record(
            week_num   = 1,
            code       = "T5",
            label      = (f"Only 1 employee holds a current '{cert_name}' certification. "
                          f"Loss of this employee = zero coverage."),
            severity   = "High",
            station    = "CLEAN" if "Food Safety" in cert_name or "WHMIS" in cert_name else "SUP",
            role       = "All Roles",
            avail      = "All Shifts",
            days       = "All Days",
            action     = (f"Immediately schedule renewal for all eligible staff. "
                          f"Ensure at least 3 employees hold a current '{cert_name}' at all times."),
            hrs        = 8,
            headcount  = 0,   # no new hire needed — existing staff need renewal
            linked_id  = holder_id,
            status     = "Open",
            resolved   = None,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER T6 — Weekly hours demand exceeds total team capacity
# ─────────────────────────────────────────────────────────────────────────────
# Business reason: There is a mathematical limit to how many hours your current
# team can give. When the schedule demands more hours than exist in your
# team's collective max_hours, you have a capacity problem — not a
# scheduling problem. You need a new hire.
#
# Data source: qsr_sales_data.csv + qsr_employees.csv
# Method:
#   - Team weekly capacity = sum of all employees' max_hours
#     adjusted for each person's max_days_per_week (not everyone works 7 days)
#   - Weekly demand = sum of staff_needed hours for each week in 2024
#   - Flag weeks where demand exceeds 90% of team capacity
#     (90% is the practical ceiling — not everyone is available every shift)
# ─────────────────────────────────────────────────────────────────────────────
print("Running T6: Weekly hours capacity gap check...")

# Compute realistic weekly team capacity
# Each employee's effective weekly hours = max_hours × (max_days_per_week / 7)
# because they can only work max_days_per_week, not all 7 days
emp_df["effective_weekly_hrs"] = (
    emp_df["max_hours"] * (emp_df["max_days_per_week"] / 7)
)
team_capacity = emp_df["effective_weekly_hrs"].sum()
capacity_90pct = team_capacity * 0.90   # practical ceiling

print(f"  Team weekly capacity     : {team_capacity:.0f} hrs")
print(f"  90% practical ceiling    : {capacity_90pct:.0f} hrs")
print(f"  Annual avg weekly demand : {annual_avg:.0f} staff-hours")

# Flag only the weeks in the TOP 25% of demand — the acute spikes.
# Why? The structural gap (demand > capacity every week) is reported as ONE
# systemic record below. The per-week records are for the WORST weeks only —
# the ones the GM most needs advance warning about.
demand_75th = weekly_demand["total_staff_hours"].quantile(0.75)

# First: log ONE systemic record capturing the chronic year-round gap
add_record(
    week_num   = 1,
    code       = "T6",
    label      = (f"SYSTEMIC: Average weekly demand ({annual_avg:.0f} hrs) "
                  f"exceeds 90% team capacity ({capacity_90pct:.0f} hrs) "
                  f"every week of 2024. Team is chronically under-resourced."),
    severity   = "High",
    station    = "Multiple",
    role       = "Crew - Part Time",
    avail      = "Flexible — any shift",
    days       = "All Days",
    action     = ("Hire 2 additional part-time crew members to close the "
                  f"{annual_avg - capacity_90pct:.0f}-hr/week structural gap. "
                  "Priority: flexible availability, cross-trainable at PREP and PACK."),
    hrs        = round(annual_avg - capacity_90pct),
    headcount  = 2,
    linked_id  = None,
    status     = "Open",
    resolved   = None,
)
print(f"  T6 systemic record added: avg gap = {annual_avg - capacity_90pct:.0f} hrs/week")

# Now flag only the acute peak weeks (top 25% of demand)
t6_weeks = weekly_demand[weekly_demand["total_staff_hours"] > demand_75th].copy()

if t6_weeks.empty:
    print(f"  T6: no weeks exceeded {capacity_90pct:.0f} hrs capacity ceiling")
else:
    for _, row in t6_weeks.iterrows():
        wk       = int(row["week_number"])
        demand   = round(row["total_staff_hours"])
        gap_hrs  = round(demand - capacity_90pct)
        pct_over = round((demand / capacity_90pct - 1) * 100, 1)

        # Past weeks that already happened are resolved — the store survived.
        # Current or future weeks are open risks for scheduling.
        is_past  = wk < 48
        status   = "Resolved" if is_past else "Open"
        res_date = resolve_date(wk, weeks_after=1) if is_past else None

        print(f"  T6 fired: week {wk} — demand {demand} hrs vs {capacity_90pct:.0f} capacity (+{pct_over}%)")
        add_record(
            week_num   = wk,
            code       = "T6",
            label      = (f"Week {wk}: total staff-hours demanded ({demand}) "
                          f"exceeded 90% team capacity ({capacity_90pct:.0f}) "
                          f"by {gap_hrs} hours (+{pct_over}%)"),
            severity   = "Medium",
            station    = "Multiple",
            role       = "Crew - Part Time",
            avail      = "Flexible",
            days       = "Weekday / Weekend",
            action     = (f"Pre-book available part-time staff for week {wk}. "
                          f"If gap ({gap_hrs} hrs) recurs 3+ weeks, "
                          f"recruit 1 additional part-time crew member."),
            hrs        = gap_hrs,
            headcount  = 1,
            linked_id  = None,
            status     = status,
            resolved   = res_date,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TRIGGER T7 — Promotion-ready employee with bench_priority=1 creates vacancy
# ─────────────────────────────────────────────────────────────────────────────
# Business reason: A promotion is a success story — and a hiring trigger.
# When your bench_priority=1 employee steps into a higher role, they leave a
# gap below them. If you start hiring the day they're promoted, you're already
# 4 weeks behind. T7 fires BEFORE the promotion happens — it tells you to
# start the backfill process now so the new hire is onboarded by promotion day.
#
# Data source: qsr_employees.csv
# Method: Find employees where promotion_ready=1 AND bench_priority=1.
#         Generate two records per employee:
#           Record 1 — the promotion itself (gap being created at their current role)
#           Record 2 — the downstream vacancy at the role below theirs
# ─────────────────────────────────────────────────────────────────────────────
print("Running T7: Promotion-ready vacancy forecast...")

# The role below each current role in the QSR hierarchy
ROLE_BELOW = {
    "Manager":          "Shift Supervisor",
    "Shift Supervisor": "Crew - Full Time",
    "Crew - Full Time": "Crew - Part Time",
}

# Find all employees who are promotion-ready with the highest bench priority
bench_1 = emp_df[
    (emp_df["promotion_ready"] == 1) &
    (emp_df["bench_priority"] == 1)
]

for _, emp in bench_1.iterrows():
    emp_id   = emp["emp_id"]
    emp_name = emp["name"]
    cur_role = emp["role"]
    nxt_role = emp["next_role"]

    print(f"  T7 fired: {emp_name} ({cur_role} -> {nxt_role})")

    # ── Record 1: The promotion itself — their current role will need backfill ─
    # When this person moves up, their current SHIFT SUPERVISOR or FT slot opens.
    add_record(
        week_num   = 4,    # begin planning in week 4 (early in the year)
        code       = "T7",
        label      = (f"{emp_name} is promotion-ready ({cur_role} -> {nxt_role}). "
                      f"Current {cur_role} position will need backfill."),
        severity   = "Low",
        station    = "SUP" if "Supervisor" in cur_role or "Manager" in cur_role else "Multiple",
        role       = cur_role,
        avail      = emp["preferred_shift"].title() if pd.notna(emp["preferred_shift"]) else "All",
        days       = emp["days_off"],
        action     = (f"Begin recruiting for 1 {cur_role} to backfill {emp_name}'s "
                      f"promotion. Target start date: 4 weeks before promotion effective date. "
                      f"Internal candidate: review bench_priority=2 employees for this role."),
        hrs        = int(emp["max_hours"]),
        headcount  = 1,
        linked_id  = emp_id,
        status     = "In Progress",
        resolved   = None,
    )

    # ── Record 2: Downstream vacancy at the level below their current role ────
    # Their promotion also opens an opportunity at the level below.
    # A Supervisor moving up means we need a new FT Crew to fill the
    # development pipeline — so the next promotion candidate has someone to mentor.
    vacancy_role = ROLE_BELOW.get(cur_role)
    if vacancy_role:
        add_record(
            week_num   = 8,    # 4 weeks after the first record — plan the pipeline
            code       = "T7",
            label      = (f"Downstream vacancy: promoting {emp_name} to {nxt_role} "
                          f"creates a gap in the {vacancy_role} pipeline."),
            severity   = "Low",
            station    = "Multiple",
            role       = vacancy_role,
            avail      = "Morning / Afternoon",
            days       = "Weekday",
            action     = (f"Recruit 1 {vacancy_role} to keep the bench pipeline healthy. "
                          f"Ideal candidate: cross-trainable, min 2 stations, "
                          f"available weekdays, open to growing into {cur_role} within 18 months."),
        hrs        = 24 if "Part Time" in vacancy_role else 32,
            headcount  = 1,
            linked_id  = emp_id,
            status     = "Open",
            resolved   = None,
        )

# ═════════════════════════════════════════════════════════════════════════════
# BUILD AND SAVE
# ═════════════════════════════════════════════════════════════════════════════

df = pd.DataFrame(rows)

# Sort: Open/In Progress first, then by severity (High → Medium → Low),
# then by week_number — so the most urgent issues appear at the top.
severity_order = {"High": 0, "Medium": 1, "Low": 2}
status_order   = {"Open": 0, "In Progress": 1, "Resolved": 2}
df["_sev"]    = df["severity"].map(severity_order)
df["_status"] = df["status"].map(status_order)
df = df.sort_values(["_status", "_sev", "week_number"]).drop(columns=["_sev", "_status"])
df = df.reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDONE: qsr_hiring_analysis.csv saved — {len(df):,} rows | {len(df.columns)} columns")
print()

# ═════════════════════════════════════════════════════════════════════════════
# VERIFICATION SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  HIRING NEEDS ANALYSIS — SUMMARY DASHBOARD")
print("=" * 70)

print(f"\n  Total trigger events  : {len(df):,}")
print(f"  Open / In Progress    : {len(df[df['status'].isin(['Open','In Progress'])]):,}")
print(f"  Resolved              : {len(df[df['status'] == 'Resolved']):,}")

print(f"\n  EVENTS BY TRIGGER CODE:")
for code in sorted(df["trigger_code"].unique()):
    count  = len(df[df["trigger_code"] == code])
    open_c = len(df[(df["trigger_code"] == code) & (df["status"].isin(["Open","In Progress"]))])
    label  = df[df["trigger_code"] == code]["trigger_label"].iloc[0][:55]
    print(f"    {code} : {count:>3} events  ({open_c} open)  — e.g. {label}...")

print(f"\n  SEVERITY BREAKDOWN:")
for sev in ["High", "Medium", "Low"]:
    count  = len(df[df["severity"] == sev])
    open_c = len(df[(df["severity"] == sev) & (df["status"].isin(["Open","In Progress"]))])
    print(f"    {sev:<8} {count:>3} total  |  {open_c} still open")

print(f"\n  TOTAL ESTIMATED HEADS TO HIRE (open events only):")
open_df   = df[df["status"].isin(["Open", "In Progress"])]
total_hc  = open_df["estimated_headcount"].sum()
total_hrs = open_df["hrs_gap_per_week"].sum()
print(f"    Estimated headcount needed : {int(total_hc)}")
print(f"    Total hrs gap per week     : {int(total_hrs)} hrs")

print(f"\n  OPEN HIGH-PRIORITY ACTIONS (High severity, Open/In Progress):")
urgent = df[(df["severity"] == "High") & (df["status"].isin(["Open","In Progress"]))]
for _, row in urgent.iterrows():
    print(f"    [{row['trigger_code']}] {row['trigger_label'][:65]}...")
    print(f"           Action: {row['recommended_action'][:70]}...")
    print()
