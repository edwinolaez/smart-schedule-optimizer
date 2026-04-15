"""
Stage 2b — Generate Compliance Training Records
================================================
WHY THIS MATTERS (Business Reason):
  Station training (Stage 2) tracks what your crew knows about the job.
  Compliance training tracks what the LAW requires them to hold.

  In Alberta (and Canada broadly), a QSR operator is legally obligated to
  ensure every person on the floor holds certain certifications before
  they're allowed to handle food, work with chemicals, or supervise others.
  Failure to maintain these is a health inspection violation — and in a
  worst case, it shuts you down.

  This dataset covers the 6 compliance certifications tracked by this store:
    1. Food Safety Level 1    — mandatory for ALL food handlers (AHS)
    2. WHMIS 2015             — mandatory for ALL staff (Gov Canada)
    3. Allergen Awareness     — mandatory for ALL staff (company policy)
    4. First Aid Level 1      — required for all managers and supervisors
    5. OH&S Fundamentals      — required for managers, supervisors, senior FT
    6. Fire Safety Awareness  — required for managers (annual)

  Module 2's compliance dashboard will read this file to:
    - Show who is currently certified
    - Highlight who is expiring within 60 days ("renewal risk")
    - Flag who is already expired ("compliance gap")
    - Generate automatic renewal reminders before a cert lapses

WHAT THIS SCRIPT DOES:
  1. Defines 6 compliance certifications with their issuing bodies and
     expiry rules (which employees need them and how long they last)
  2. Generates one record per employee × applicable certification
  3. Spreads issued dates realistically across 2019-2024 based on cert type
  4. Calculates expiry dates, days until expiry, and renewal status
     relative to December 31 2024 (our data snapshot date)
  5. Deliberately makes some certs expired or expiring soon for realism
     (people procrastinate on renewals — this is real-world accurate)
  6. Saves to data/qsr_compliance_training.csv

THE 9 COLUMNS:
  record_id          — unique ID (CR001, CR002 ...)
  employee_id        — matches emp_id in qsr_employees.csv
  certification_name — name of the compliance certification
  issued_date        — date the certification was issued (YYYY-MM-DD)
  expiry_date        — date the certification expires (YYYY-MM-DD)
  is_expired         — 1 if expiry_date is before Dec 31 2024, else 0
  days_until_expiry  — days from Dec 31 2024 to expiry_date
                       (negative = already expired, positive = days left)
  renewal_status     — Current / Expiring Soon / Expired
  issuing_body       — organization that issues the certification

Outputs:
  data/qsr_compliance_training.csv   (~59 rows, 9 columns)
"""

import pandas as pd                           # our spreadsheet tool
import numpy as np                            # math and random numbers
import os                                     # file path builder
from datetime import date, timedelta          # working with calendar dates
import random                                 # picking random dates and values

# ── Random seeds ──────────────────────────────────────────────────────────────
np.random.seed(33)
random.seed(33)

# ── File paths ────────────────────────────────────────────────────────────────
BASE        = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EMP_PATH    = os.path.join(BASE, "data", "qsr_employees.csv")
OUTPUT_PATH = os.path.join(BASE, "data", "qsr_compliance_training.csv")

# ── Load employees ────────────────────────────────────────────────────────────
print("Loading employee roster...")
emp_df = pd.read_csv(EMP_PATH)
print(f"  {len(emp_df)} employees loaded")

# ── Snapshot date ─────────────────────────────────────────────────────────────
# This is the date we use as "today" when calculating days_until_expiry.
# All data in this platform represents the state of the store at end of 2024.
SNAPSHOT = date(2024, 12, 31)

# ── Compliance certification definitions ──────────────────────────────────────
# Each entry defines one type of compliance cert.
#
# applies_to:  "all" means every employee needs it,
#              or a list of emp_ids who specifically require it
#
# expiry_days: how long the cert is valid from issued_date
#
# issued_window: (earliest, latest) dates we'd expect these to have been issued
#   - Long-lived certs (5yr Food Safety) were issued 2019-2022
#   - Annual certs (WHMIS, Allergen) were mostly issued in 2024, some 2023
#   - Mid-length certs (First Aid, 3yr) were issued 2022-2024
#
# force_expired_ids: specific employees whose cert we intentionally set as
#   expired — realistic because people delay renewals, especially part-timers

CERTIFICATIONS = [
    {
        # All food handlers in Alberta must hold this. Issued by Alberta Health
        # Services. Valid for 5 years. This is the most important one —
        # a health inspector will check this first.
        "name":         "Food Safety Level 1",
        "issuing_body": "Alberta Health Services",
        "expiry_days":  5 * 365,    # 5-year cert
        "applies_to":   "all",
        "issued_window": (date(2019, 6, 1), date(2022, 12, 31)),
        "force_expired_ids": ["E009"],  # Fatima is overdue for renewal
    },
    {
        # Workplace Hazardous Materials Information System — federal requirement
        # for anyone who may encounter chemical products at work.
        # In a QSR that means cleaning products, sanitizers, fryer chemicals.
        # Annual renewal is best practice (technically every 3 years but
        # most operators enforce annual given health inspection scrutiny).
        "name":         "WHMIS 2015",
        "issuing_body": "Government of Canada / CCOHS",
        "expiry_days":  365,        # annual renewal
        "applies_to":   "all",
        "issued_window": (date(2023, 6, 1), date(2024, 9, 30)),
        "force_expired_ids": ["E010", "E013", "E015"],  # PT crew who missed renewal
    },
    {
        # Company-required allergen awareness training. Every person who
        # handles food must know the 14 major allergens and cross-contamination
        # risks. A customer allergic to peanuts doesn't care that the cook
        # "didn't know" — and neither will the coroner.
        "name":         "Allergen Awareness",
        "issuing_body": "QSR Internal Training (Company Policy)",
        "expiry_days":  365,        # renewed annually with food safety refresher
        "applies_to":   "all",
        "issued_window": (date(2023, 9, 1), date(2024, 11, 30)),
        "force_expired_ids": [],    # we keep this one clean for contrast
    },
    {
        # Required for all managers and supervisors. Standard in any
        # workplace where you're responsible for others' safety.
        # Canadian Red Cross Level 1 covers CPR + AED + basic first aid.
        # Valid for 3 years.
        "name":         "First Aid Level 1",
        "issuing_body": "Canadian Red Cross",
        "expiry_days":  3 * 365,    # 3-year cert
        "applies_to":   ["E001", "E002", "E003", "E004", "E005"],
        "issued_window": (date(2021, 6, 1), date(2024, 6, 30)),
        "force_expired_ids": ["E004"],  # Kevin Park needs to renew
    },
    {
        # Alberta Occupational Health & Safety — required for anyone who
        # may be asked to supervise a shift, conduct safety walkthroughs,
        # or onboard new employees under AB OHS legislation.
        # Applies to managers, supervisors, and senior full-time crew
        # who are in the succession pipeline (promotion_ready = 1).
        "name":         "OH&S Fundamentals",
        "issuing_body": "Government of Alberta",
        "expiry_days":  3 * 365,    # 3-year, refreshed with regulation changes
        "applies_to":   ["E001", "E002", "E003", "E004", "E005", "E006", "E008"],
        "issued_window": (date(2020, 1, 1), date(2023, 12, 31)),
        "force_expired_ids": [],
    },
    {
        # Annual fire safety awareness required for store managers.
        # Covers evacuation procedures, extinguisher use, fryer fire response.
        # Issued by the city fire prevention office or equivalent authority.
        "name":         "Fire Safety Awareness",
        "issuing_body": "City Fire Prevention Office",
        "expiry_days":  365,        # annual renewal
        "applies_to":   ["E001", "E002"],
        "issued_window": (date(2024, 1, 1), date(2024, 10, 31)),
        "force_expired_ids": [],
    },
]

# ── Date helper ───────────────────────────────────────────────────────────────
def random_date(start: date, end: date) -> date:
    """Pick a random date between start and end (inclusive)."""
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))

# ── Renewal status label ──────────────────────────────────────────────────────
# Like a traffic light on your compliance dashboard:
#   Green  = Current          (more than 60 days remaining)
#   Yellow = Expiring Soon    (within 60 days — schedule renewal now)
#   Red    = Expired          (already lapsed — stop the clock, this is a risk)
def get_renewal_status(days_left: int, is_exp: int) -> str:
    if is_exp == 1:
        return "Expired"
    elif days_left <= 60:
        return "Expiring Soon"
    else:
        return "Current"

# ── Generate compliance records ───────────────────────────────────────────────
print("Generating compliance records...")
rows           = []
record_counter = 0

for cert in CERTIFICATIONS:

    # Determine which employees this cert applies to
    if cert["applies_to"] == "all":
        # Every employee in the roster needs this one
        applies_to_ids = emp_df["emp_id"].tolist()
    else:
        # Only specific employees (by emp_id)
        applies_to_ids = cert["applies_to"]

    for emp_id in applies_to_ids:

        is_forced_expired = emp_id in cert["force_expired_ids"]

        # ── Generate the issued_date ──────────────────────────────────────────
        if is_forced_expired:
            # Force an old issue date so the cert is definitely expired.
            # We back-calculate: expiry = issued + expiry_days.
            # For it to be expired by SNAPSHOT, issued must be more than
            # expiry_days before SNAPSHOT.
            # We add a small buffer (30-180 days) so it's clearly lapsed.
            extra_lapse_days = random.randint(30, 180)
            issued_date = SNAPSHOT - timedelta(days=cert["expiry_days"] + extra_lapse_days)
        else:
            # Normal case: pick a date within the defined issued window
            issued_date = random_date(cert["issued_window"][0], cert["issued_window"][1])

        # ── Calculate expiry_date ─────────────────────────────────────────────
        expiry_date = issued_date + timedelta(days=cert["expiry_days"])

        # ── is_expired and days_until_expiry ──────────────────────────────────
        # days_until_expiry is negative if already expired — like a countdown
        # that went past zero. A negative number tells you HOW overdue they are.
        is_expired       = 1 if expiry_date < SNAPSHOT else 0
        days_until_expiry = (expiry_date - SNAPSHOT).days   # negative = overdue

        renewal_status = get_renewal_status(days_until_expiry, is_expired)

        # ── Build the row ─────────────────────────────────────────────────────
        record_counter += 1
        rows.append({
            "record_id":          f"CR{record_counter:03d}",
            "employee_id":        emp_id,
            "certification_name": cert["name"],
            "issued_date":        issued_date.strftime("%Y-%m-%d"),
            "expiry_date":        expiry_date.strftime("%Y-%m-%d"),
            "is_expired":         is_expired,
            "days_until_expiry":  days_until_expiry,
            "renewal_status":     renewal_status,
            "issuing_body":       cert["issuing_body"],
        })

# ── Build DataFrame and save ──────────────────────────────────────────────────
df = pd.DataFrame(rows)

# Sort by renewal_status priority (Expired first, then Expiring Soon, then Current)
# then by employee — same order you'd want on a compliance dashboard
status_order = {"Expired": 0, "Expiring Soon": 1, "Current": 2}
df["_sort_key"] = df["renewal_status"].map(status_order)
df = df.sort_values(["_sort_key", "employee_id", "certification_name"]).drop(columns="_sort_key")
df = df.reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDONE: qsr_compliance_training.csv saved — {len(df):,} rows | {len(df.columns)} columns")
print()

# ── Verification summary ──────────────────────────────────────────────────────
print("=" * 65)
print("  COMPLIANCE TRAINING SUMMARY — snapshot: 2024-12-31")
print("=" * 65)

print(f"\n  Total records     : {len(df):,}")
print(f"  Employees covered : {df['employee_id'].nunique()}")
print(f"  Cert types        : {df['certification_name'].nunique()}")

# ── Status breakdown — the compliance dashboard top-line ─────────────────────
print(f"\n  RENEWAL STATUS BREAKDOWN:")
for status in ["Expired", "Expiring Soon", "Current"]:
    count  = len(df[df["renewal_status"] == status])
    pct    = count / len(df) * 100
    marker = "  <-- ACTION REQUIRED" if status == "Expired" else (
             "  <-- SCHEDULE RENEWAL" if status == "Expiring Soon" else "")
    print(f"    {status:<16} {count:>3} records ({pct:.0f}%){marker}")

# ── Expired certifications — these are compliance GAPS ───────────────────────
expired = df[df["is_expired"] == 1]
print(f"\n  EXPIRED CERTIFICATIONS ({len(expired)}) — immediate action needed:")
emp_names = emp_df[["emp_id", "name"]].rename(columns={"emp_id": "employee_id"})
if not expired.empty:
    exp_detail = expired.merge(emp_names, on="employee_id")
    for _, row in exp_detail.iterrows():
        overdue_days = abs(row["days_until_expiry"])
        print(f"    {row['employee_id']}  {row['name']:<22}  "
              f"{row['certification_name']:<28}  {overdue_days} days overdue")

# ── Expiring soon — upcoming renewal risk ─────────────────────────────────────
expiring_soon = df[df["renewal_status"] == "Expiring Soon"]
print(f"\n  EXPIRING WITHIN 60 DAYS ({len(expiring_soon)}) — schedule renewals:")
if not expiring_soon.empty:
    exp_s_detail = expiring_soon.merge(emp_names, on="employee_id")
    for _, row in exp_s_detail.iterrows():
        print(f"    {row['employee_id']}  {row['name']:<22}  "
              f"{row['certification_name']:<28}  {row['days_until_expiry']} days left  "
              f"(expires {row['expiry_date']})")
else:
    print("    None expiring within 60 days.")

# ── Coverage per certification type ──────────────────────────────────────────
print(f"\n  ACTIVE COVERAGE PER CERTIFICATION TYPE:")
print(f"  (employees with Current or Expiring Soon status)")
for cert_name in df["certification_name"].unique():
    cert_df     = df[df["certification_name"] == cert_name]
    total_req   = len(cert_df)
    active      = len(cert_df[cert_df["renewal_status"].isin(["Current", "Expiring Soon"])])
    expired_ct  = len(cert_df[cert_df["renewal_status"] == "Expired"])
    gap_flag    = "  <-- COMPLIANCE GAP" if expired_ct > 0 else ""
    print(f"    {cert_name:<32} {active}/{total_req} compliant{gap_flag}")

# ── Per-employee compliance score ────────────────────────────────────────────
print(f"\n  EMPLOYEE COMPLIANCE OVERVIEW:")
emp_compliance = (
    df.groupby("employee_id")
    .apply(lambda g: {
        "total":   len(g),
        "current": (g["renewal_status"] == "Current").sum(),
        "soon":    (g["renewal_status"] == "Expiring Soon").sum(),
        "expired": (g["is_expired"] == 1).sum(),
    }, include_groups=False)
    .reset_index()
)
# Flatten the dict column
emp_compliance = pd.DataFrame(
    emp_compliance[0].tolist(),
    index=emp_compliance["employee_id"]
).reset_index()
emp_compliance = emp_compliance.merge(emp_names, on="employee_id")

for _, row in emp_compliance.sort_values("expired", ascending=False).iterrows():
    gap = "  <-- HAS GAPS" if row["expired"] > 0 else ""
    print(f"    {row['employee_id']}  {row['name']:<22}  "
          f"Current:{int(row['current'])}  "
          f"Soon:{int(row['soon'])}  "
          f"Expired:{int(row['expired'])}{gap}")
