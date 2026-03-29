"""
Stage 3 -- Schedule Optimizer (Enhanced)
Generates a weekly staff schedule using the Stage 2b enhanced ML model.
Applies all 9 hard scheduling rules.
Includes: trend adoption, LY context, transaction projections,
          and per-day reliability scores.

Usage:
  python stage3_optimizer.py                    # week of 2024-01-08 (3 weeks ahead)
  python stage3_optimizer.py 2024-03-04         # any Monday, defaults to 3 weeks ahead
  python stage3_optimizer.py 2024-03-04 1       # 1 week ahead (higher reliability)

Outputs:
  data/schedule_output.csv
  reports/schedule_week_YYYY-MM-DD.txt
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import date, timedelta

BASE          = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH    = os.path.join(BASE, "models",  "staff_model_enhanced.pkl")
EMP_PATH      = os.path.join(BASE, "data",    "qsr_employees.csv")
LY_AGG_PATH   = os.path.join(BASE, "models",  "ly_agg_2023.csv")
WH_PATH       = os.path.join(BASE, "models",  "weekly_hourly_2024.csv")
OUTPUT_CSV    = os.path.join(BASE, "data",    "schedule_output.csv")

# ── Growth factor toggle ──────────────────────────────────────────────────────
# Set APPLY_GROWTH = True to activate 5% YoY growth on all projections.
# When False, predictions use raw LY + trend data with no growth adjustment.
APPLY_GROWTH  = False
GROWTH_FACTOR = 1.05  # 5% YoY -- only used when APPLY_GROWTH is True

SHIFTS = {
    "OPEN":   (6,  14, 8),
    "MID":    (10, 18, 8),
    "CLOSE":  (15, 23, 8),
    "PT_AM":  (6,  12, 6),
    "PT_PM":  (12, 18, 6),
    "PT_EVE": (17, 23, 6),
}

SUPERVISOR_ROLES = {"Manager", "Shift Supervisor"}
DAY_ORDER        = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEATHER_MAP      = {"sunny": 0, "cloudy": 1, "rainy": 2, "snowy": 3}
SEASONAL_WEATHER = {
    1:"snowy", 2:"snowy", 3:"cloudy", 4:"rainy",
    5:"sunny", 6:"sunny", 7:"sunny",  8:"sunny",
    9:"cloudy",10:"rainy",11:"cloudy",12:"snowy",
}

# ── Alberta holidays (for any year) ──────────────────────────────────────────
def is_holiday(d: date) -> bool:
    yr = d.year
    # Fixed-date holidays
    fixed = {(1,1),(7,1),(11,11),(12,25),(12,26)}
    if (d.month, d.day) in fixed:
        return True
    # Family Day: 3rd Monday of February
    mondays = [date(yr,2,i) for i in range(1,29) if date(yr,2,i).weekday()==0]
    if len(mondays) >= 3 and d == mondays[2]:
        return True
    # Victoria Day: last Monday before May 25
    for day in range(24,18,-1):
        try:
            vd = date(yr,5,day)
            if vd.weekday() == 0:
                if d == vd: return True
                break
        except: pass
    # Heritage Day: 1st Monday of August
    for day in range(1,8):
        hd = date(yr,8,day)
        if hd.weekday() == 0:
            if d == hd: return True
            break
    # Labour Day: 1st Monday of September
    for day in range(1,8):
        ld = date(yr,9,day)
        if ld.weekday() == 0:
            if d == ld: return True
            break
    # Thanksgiving: 2nd Monday of October
    mondays_oct = [date(yr,10,i) for i in range(1,32) if date(yr,10,i).weekday()==0]
    if len(mondays_oct) >= 2 and d == mondays_oct[1]:
        return True
    return False

# ── Load model bundle + lookup tables ────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model    = bundle["model"]
FEATURES = bundle["features"]

ly_agg        = pd.read_csv(LY_AGG_PATH)
weekly_hourly = pd.read_csv(WH_PATH)
employees     = pd.read_csv(EMP_PATH)
employees["days_off_list"] = employees["days_off"].apply(
    lambda s: [d.strip() for d in str(s).split(",")]
)

# ── LY lookup helpers ─────────────────────────────────────────────────────────
def get_ly_same_week(iso_week, hour):
    row = ly_agg[(ly_agg["iso_week"]==iso_week) & (ly_agg["hour"]==hour)]
    if row.empty:
        nearby = ly_agg[
            (ly_agg["iso_week"].between(iso_week-1, iso_week+1)) &
            (ly_agg["hour"]==hour)
        ]
        if nearby.empty: return 0.0, 0.0, 0.0
        return nearby["ly_cust_mean"].mean(), nearby["ly_txn_mean"].mean(), nearby["ly_cust_std"].mean()
    return float(row["ly_cust_mean"].iloc[0]), float(row["ly_txn_mean"].iloc[0]), float(row["ly_cust_std"].iloc[0])

def get_ly_context(iso_week, hour, window=3):
    weeks = list(range(max(1, iso_week-window), min(53, iso_week+window+1)))
    ctx = ly_agg[(ly_agg["iso_week"].isin(weeks)) & (ly_agg["hour"]==hour)]
    if ctx.empty: return 0.0, 0.0
    return ctx["ly_cust_mean"].mean(), ctx["ly_txn_mean"].mean()

def get_trend_ratio(iso_week, hour):
    recent_weeks = list(range(max(1, iso_week-3), iso_week))
    prior_weeks  = list(range(max(1, iso_week-6), max(1, iso_week-3)))
    def wk_avg(wks):
        r = weekly_hourly[(weekly_hourly["iso_week"].isin(wks)) & (weekly_hourly["hour"]==hour)]
        return r["weekly_avg"].mean() if not r.empty else None
    ra = wk_avg(recent_weeks)
    pa = wk_avg(prior_weeks)
    if ra and pa and pa > 0 and not np.isnan(ra) and not np.isnan(pa):
        return round(float(ra)/float(pa), 4)
    return 1.0

def reliability_score(iso_week, hour, weeks_ahead):
    _, _, ly_std  = get_ly_same_week(iso_week, hour)
    ly_mean, _    = get_ly_context(iso_week, hour)
    cv = (ly_std / ly_mean) if ly_mean > 0 else 0.15
    cv = min(cv, 0.40)
    base  = max(0.50, min(0.95, 1.0 - cv * 1.5))
    score = round(base * (0.97 ** weeks_ahead), 3)
    pct   = f"{score*100:.0f}%"
    label = "HIGH" if score >= 0.85 else ("MEDIUM" if score >= 0.72 else "LOW")
    return score, pct, label

# ── Predict hourly staffing for a day ─────────────────────────────────────────
# ── Event presets — default traffic multipliers ───────────────────────────────
EVENT_PRESETS = {
    "None":                        1.00,
    "School Day Off / PA Day":     1.15,
    "School Break Week":           1.20,
    "Community Event Nearby":      1.25,
    "Local Festival / Parade":     1.30,
    "Major Sporting Event":        1.20,
    "Construction / Road Closure": 0.80,
    "Nearby Competitor Closed":    1.15,
    "Custom":                      1.00,
}

def predict_day(target_date: date, weeks_ahead: int,
                weather: str = None, is_promotion: int = 0,
                event_multiplier: float = 1.0) -> dict:
    """
    Returns per-hour dict with:
      staff_raw, staff_growth, customer_proj, txn_proj, reliability

    Args:
      weather          -- override forecast: "sunny"|"cloudy"|"rainy"|"snowy"
                          defaults to seasonal monthly average if None
      is_promotion     -- 1 if store is running a promotion that day, else 0
      event_multiplier -- traffic adjustment factor (e.g. 1.25 = +25% for a
                          nearby community event, 0.80 = -20% for construction)
                          Applied on top of ML prediction and LY projections.
    """
    day_name   = target_date.strftime("%A")
    day_n      = DAY_ORDER.index(day_name)
    is_weekend = 1 if target_date.weekday() >= 5 else 0
    is_hol     = 1 if is_holiday(target_date) else 0
    iso_week   = target_date.isocalendar()[1]
    weather    = weather if weather in WEATHER_MAP else SEASONAL_WEATHER[target_date.month]
    weather_n  = WEATHER_MAP[weather]

    results = {}
    for hour in range(6, 24):
        ly_cust, ly_txn, ly_std     = get_ly_same_week(iso_week, hour)
        ctx_cust, ctx_txn           = get_ly_context(iso_week, hour)
        trend                       = get_trend_ratio(iso_week, hour)
        rel_score, rel_pct, rel_lbl = reliability_score(iso_week, hour, weeks_ahead)

        X = pd.DataFrame([[
            hour, day_n, is_weekend, is_hol, is_promotion, weather_n,
            ly_cust, ly_txn, ctx_cust, ctx_txn, trend
        ]], columns=FEATURES)

        gf = GROWTH_FACTOR if APPLY_GROWTH else 1.0

        # ML base prediction
        staff_base = max(2, int(np.round(model.predict(X)[0])))
        # Apply event multiplier on top of ML base
        staff_raw    = max(2, int(np.ceil(staff_base * event_multiplier)))
        staff_growth = max(2, int(np.ceil(staff_raw  * gf)))

        # Forward projections (LY × trend × growth × event)
        cust_proj = int(ly_cust * gf * trend * event_multiplier)
        txn_proj  = int(ly_txn  * gf * trend * event_multiplier)

        results[hour] = {
            "staff_base":   staff_base,   # pure ML, no event adjustment
            "staff_raw":    staff_raw,    # ML + event multiplier
            "staff_growth": staff_growth,
            "cust_proj":    cust_proj,
            "txn_proj":     txn_proj,
            "trend":        trend,
            "rel_score":    rel_score,
            "rel_pct":      rel_pct,
            "rel_label":    rel_lbl,
        }
    return results

# ── Shift plan from peak demand ───────────────────────────────────────────────
def build_shift_plan(hourly: dict) -> list:
    peak_am  = max(hourly[h]["staff_growth"] for h in range(6,  12))
    peak_mid = max(hourly[h]["staff_growth"] for h in range(11, 15))
    peak_eve = max(hourly[h]["staff_growth"] for h in range(18, 24))

    plan = []
    n_open  = max(2, int(np.ceil(peak_mid / 3)))
    n_close = max(2, int(np.ceil(peak_eve / 3)))

    plan += ["OPEN"]  * n_open
    if peak_mid >= 5:
        plan += ["MID"] * max(1, int(np.ceil((peak_mid - n_open*2) / 3)))
    plan += ["CLOSE"] * n_close

    am_gap  = max(0, peak_am  - n_open * 2)
    eve_gap = max(0, peak_eve - n_close * 2)
    if am_gap  > 0: plan += ["PT_AM"]  * min(am_gap,  2)
    if eve_gap > 0: plan += ["PT_EVE"] * min(eve_gap, 2)

    return plan

# ── Hard rules ────────────────────────────────────────────────────────────────
def rule1(emp, day_name):        return day_name not in emp["days_off_list"]
def rule3(emp, state, hrs):      return state[emp["emp_id"]]["hours_worked"] + hrs <= emp["max_hours"]
def rule5(emp, start, end):      return start >= emp["availability_start"] and end <= emp["availability_end"]
def rule6(emp, state, start, d):
    last = state[emp["emp_id"]]["last_shift_end"]
    if last is None: return True
    last_date, last_hour = last
    gap = (d - last_date).days * 24 + (start - last_hour)
    return gap >= emp["min_rest_hours"]
def rule7(emp, state):           return state[emp["emp_id"]]["days_worked"] < emp["max_days_per_week"]

def assign_employee(shift_label, day_date, day_name, emp_state, emps, assigned_today, need_sup):
    start, end, hrs = SHIFTS[shift_label]
    candidates = []
    for _, emp in emps.iterrows():
        eid = emp["emp_id"]
        if eid in assigned_today:              continue
        if not rule1(emp, day_name):           continue
        if not rule5(emp, start, end):         continue
        if not rule3(emp, emp_state, hrs):     continue
        if not rule6(emp, emp_state, start, day_date): continue
        if not rule7(emp, emp_state):          continue
        is_sup = emp["role"] in SUPERVISOR_ROLES
        hours_remaining = emp["max_hours"] - emp_state[eid]["hours_worked"]
        candidates.append((emp, is_sup, hours_remaining))
    if not candidates: return None
    if need_sup:
        candidates.sort(key=lambda x: (not x[1], -x[2]))
    else:
        candidates.sort(key=lambda x: (x[1], -x[2]))
    emp = candidates[0][0]
    return {"emp_id": emp["emp_id"], "name": emp["name"], "role": emp["role"],
            "shift_type": shift_label, "start_hour": start, "end_hour": end, "hours": hrs}

# ── Schedule full week ────────────────────────────────────────────────────────
def schedule_week(week_start: date, weeks_ahead: int,
                  day_overrides: dict = None) -> tuple:
    """
    day_overrides: optional dict keyed by date with per-day factor overrides.
      e.g. { date(2024,6,10): {"weather": "rainy", "is_promotion": 1} }
    """
    emp_state = {
        row["emp_id"]: {"days_worked": 0, "hours_worked": 0.0, "last_shift_end": None}
        for _, row in employees.iterrows()
    }

    all_assignments = []
    day_projections = {}  # date -> summary for report

    for offset in range(7):
        day_date  = week_start + timedelta(days=offset)
        day_name  = day_date.strftime("%A")
        iso_week  = day_date.isocalendar()[1]
        overrides = (day_overrides or {}).get(day_date, {})

        hourly     = predict_day(day_date, weeks_ahead,
                                 weather=overrides.get("weather"),
                                 is_promotion=overrides.get("is_promotion", 0),
                                 event_multiplier=overrides.get("event_multiplier", 1.0))
        shift_plan = build_shift_plan(hourly)

        # Day-level summary metrics
        peak_hour      = max(hourly, key=lambda h: hourly[h]["staff_growth"])
        peak_staff     = hourly[peak_hour]["staff_growth"]
        peak_cust      = hourly[peak_hour]["cust_proj"]
        peak_txn       = hourly[peak_hour]["txn_proj"]
        avg_trend      = round(np.mean([hourly[h]["trend"] for h in hourly]), 3)
        avg_rel_score  = round(np.mean([hourly[h]["rel_score"] for h in hourly]), 3)
        avg_rel_pct    = f"{avg_rel_score*100:.0f}%"
        avg_rel_label  = "HIGH" if avg_rel_score >= 0.85 else ("MEDIUM" if avg_rel_score >= 0.72 else "LOW")

        day_projections[day_date] = {
            "peak_staff":   peak_staff,
            "peak_cust":    peak_cust,
            "peak_txn":     peak_txn,
            "trend":        avg_trend,
            "rel_pct":      avg_rel_pct,
            "rel_label":    avg_rel_label,
            "iso_week":     iso_week,
            "weather":          overrides.get("weather") or SEASONAL_WEATHER[day_date.month],
            "is_promotion":     overrides.get("is_promotion", 0),
            "event_multiplier": overrides.get("event_multiplier", 1.0),
            "event_label":      overrides.get("event_label", "None"),
        }

        day_assignments    = []
        assigned_today     = set()
        supervisor_covered = False

        for shift_label in shift_plan:
            need_sup = not supervisor_covered
            result   = assign_employee(shift_label, day_date, day_name,
                                       emp_state, employees, assigned_today, need_sup)
            if result is None: continue

            eid = result["emp_id"]
            day_assignments.append(result)
            assigned_today.add(eid)
            emp_state[eid]["days_worked"]   += 1
            emp_state[eid]["hours_worked"]  += result["hours"]
            emp_state[eid]["last_shift_end"] = (day_date, result["end_hour"])
            if result["role"] in SUPERVISOR_ROLES:
                supervisor_covered = True

        for a in day_assignments:
            a.update({"date": day_date.strftime("%Y-%m-%d"), "day": day_name,
                      "peak_staff_growth": peak_staff})
        all_assignments.extend(day_assignments)

        trend_arrow = "+" if avg_trend >= 1.0 else "-"
        print(f"  {day_name:10} {day_date}  peak={peak_staff} staff "
              f"| cust~{peak_cust}/hr | txn~{peak_txn}/hr "
              f"| trend {trend_arrow}{abs(avg_trend-1)*100:.1f}% "
              f"| reliability {avg_rel_pct} [{avg_rel_label}]")

    df = pd.DataFrame(all_assignments)[[
        "date","day","emp_id","name","role",
        "shift_type","start_hour","end_hour","hours","peak_staff_growth"
    ]]
    return df, day_projections

# ── Print + save report ───────────────────────────────────────────────────────
def save_report(df, projections, week_start, weeks_ahead):
    report_path = os.path.join(BASE, "reports", f"schedule_week_{week_start}.txt")
    lines = []
    lines.append("=" * 70)
    lines.append(f"  WEEKLY SCHEDULE -- Week of {week_start}")
    growth_note = f"+{(GROWTH_FACTOR-1)*100:.0f}% YoY growth ON" if APPLY_GROWTH else "growth factor OFF (LY + trend only)"
    lines.append(f"  Generated {weeks_ahead} week(s) in advance | {growth_note}")
    lines.append("=" * 70)

    for day_date, proj in projections.items():
        day_name = day_date.strftime("%A")
        day_df   = df[df["date"] == day_date.strftime("%Y-%m-%d")].sort_values("start_hour")

        trend_str  = f"+{(proj['trend']-1)*100:.1f}%" if proj['trend'] >= 1 else f"{(proj['trend']-1)*100:.1f}%"
        ly_note    = f"(trend {trend_str} from last 3 wks)"
        rel_str    = f"Reliability: {proj['rel_pct']} [{proj['rel_label']}]  ({weeks_ahead}wk ahead)"

        lines.append(f"\n{day_name.upper()} {day_date}  ISO Week {proj['iso_week']}")
        lines.append(f"  Projection: ~{proj['peak_cust']} customers/hr peak | ~{proj['peak_txn']} txn/hr | {proj['peak_staff']} staff needed")
        lines.append(f"  {rel_str}  {ly_note}")
        lines.append("-" * 70)
        lines.append(f"  {'Name':<22} {'Role':<20} {'Shift':<8} {'Hours'}")
        lines.append(f"  {'-'*22} {'-'*20} {'-'*8} {'-'*13}")
        for _, row in day_df.iterrows():
            t = f"{row['start_hour']:02d}:00-{row['end_hour']:02d}:00"
            lines.append(f"  {row['name']:<22} {row['role']:<20} {row['shift_type']:<8} {t}  ({row['hours']}h)")
        lines.append(f"  Total staff scheduled: {len(day_df)}")

    # Weekly hours summary
    lines.append("\n" + "=" * 70)
    lines.append("  WEEKLY HOURS SUMMARY")
    lines.append("=" * 70)
    summary = (df.groupby(["emp_id","name","role"])["hours"]
               .agg(total_hours="sum", shifts="count")
               .reset_index()
               .sort_values("total_hours", ascending=False))
    lines.append(f"\n  {'ID':<6} {'Name':<22} {'Role':<20} {'Hours':>6} {'Shifts':>7}")
    lines.append(f"  {'-'*6} {'-'*22} {'-'*20} {'-'*6} {'-'*7}")
    for _, row in summary.iterrows():
        lines.append(f"  {row['emp_id']:<6} {row['name']:<22} {row['role']:<20}"
                     f" {row['total_hours']:>6.0f} {row['shifts']:>7}")

    # Projection confidence summary
    lines.append("\n" + "=" * 70)
    lines.append("  PROJECTION CONFIDENCE SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Weeks scheduled in advance : {weeks_ahead}")
    lines.append(f"  YoY growth factor applied  : {'+'+(str(int((GROWTH_FACTOR-1)*100)))+'% (ON)' if APPLY_GROWTH else 'OFF -- set APPLY_GROWTH=True to enable'}")
    lines.append(f"  LY reference data          : 2023 (same ISO week +/- 3 weeks)")
    lines.append(f"  Trend source               : last 3 weeks vs prior 3 weeks (2024)")
    lines.append(f"  Reliability decay rate     : 3% per week ahead of scheduling")
    lines.append("")
    for day_date, proj in projections.items():
        lines.append(f"  {day_date.strftime('%A'):<10} {day_date}  {proj['rel_pct']:>4} [{proj['rel_label']:<6}]")

    text = "\n".join(lines)
    print("\n" + text)
    with open(report_path, "w") as f:
        f.write(text)
    print(f"\nReport saved --> {report_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        week_start = date.fromisoformat(sys.argv[1])
        if week_start.weekday() != 0:
            print("ERROR: week_start must be a Monday.")
            sys.exit(1)
    else:
        week_start = date(2024, 1, 8)

    weeks_ahead = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Generating schedule for week of {week_start} ({weeks_ahead} week(s) ahead)\n")
    schedule_df, day_proj = schedule_week(week_start, weeks_ahead)

    schedule_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSchedule CSV saved --> {OUTPUT_CSV}")

    save_report(schedule_df, day_proj, week_start, weeks_ahead)
    print(f"\nStage 3 complete. {len(schedule_df)} shift assignments generated.")
