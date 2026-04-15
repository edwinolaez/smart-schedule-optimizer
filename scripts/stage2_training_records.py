"""
Stage 2 — Generate Station Training Records
============================================
WHY THIS MATTERS (Business Reason):
  Module 2 is Training & Development. Before you can build dashboards that show
  who needs training, who is overdue for renewal, or who is ready to become a
  trainer — you need the underlying record of every training event that has
  ever happened for every employee at every station.

  In a real restaurant, this is your "training binder" — the paper or digital
  log that tracks:
    - When each employee first learned a station (Initial Training)
    - When they passed their certification assessment (Competency Check)
    - When they reached proficiency and can be trusted at peak hours
    - When they earned the right to train others (Trainer Certification)
    - When mandatory certifications expire and need renewal (CLEAN, SUP)

  This dataset is also what feeds Module 3 hiring triggers:
    T1: Station has fewer than 2 certified staff (Level 2+)
    T5: Only 1 employee holds any role or station certification

  If you can't count how many people are certified at each station,
  you can't detect when you're dangerously understaffed at a skill level.

WHAT THIS SCRIPT DOES:
  1. Loads qsr_employees.csv to read each employee's station proficiency levels
  2. For each employee × station with proficiency = 1:
       Generates 1 record — Initial Training (in progress, not yet certified)
  3. For each employee × station with proficiency >= 2:
       Generates 2 records:
         Record A (older date) — Initial Training completed
         Record B (newer date) — the current certification level achieved
         (Competency Check for L=2, Proficiency Assessment for L=3,
          Trainer Certification for L=4)
  4. Generates 1 Annual CLEAN renewal record per employee (mandatory)
  5. Saves to data/qsr_training_records.csv

THE 15 COLUMNS:
  record_id          — unique ID (TR001, TR002 ...)
  employee_id        — matches emp_id in qsr_employees.csv
  station_code       — BEV / PREP / PACK / CLEAN / OPEN / CLOSE / SUP / TRAIN
  training_type      — Initial Training / Competency Check /
                       Proficiency Assessment / Trainer Certification /
                       Annual Renewal
  trainer_id         — employee_id of who conducted the training
  training_date      — date the training session was held
  score              — assessment score (0-100); null if not yet assessed
  proficiency_level  — level achieved by this record (1-4)
  certification_date — date certified; null if still in progress
  expiry_date        — date cert expires; null if no expiry for this type
  is_expired         — 1 if expiry_date is in the past, else 0
  followup_date      — scheduled followup date (if needed)
  followup_done      — 1 if followup was completed, else 0
  notes              — short text note from the training session
  status             — Completed / In Progress / Expired / Pending Renewal

Outputs:
  data/qsr_training_records.csv   (~220 rows, 15 columns)
"""

import pandas as pd                          # our spreadsheet tool
import numpy as np                           # math and random numbers
import os                                    # file path builder
from datetime import date, timedelta         # for working with calendar dates
import random                                # for picking trainers and notes

# ── Random seeds ─────────────────────────────────────────────────────────────
np.random.seed(21)
random.seed(21)

# ── File paths ────────────────────────────────────────────────────────────────
BASE        = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EMP_PATH    = os.path.join(BASE, "data", "qsr_employees.csv")
OUTPUT_PATH = os.path.join(BASE, "data", "qsr_training_records.csv")

# ── Load employee data ────────────────────────────────────────────────────────
print("Loading employee roster...")
emp_df = pd.read_csv(EMP_PATH)
print(f"  {len(emp_df)} employees loaded")

# ── Station definitions ───────────────────────────────────────────────────────
# The 8 station codes from the platform design.
# CLEAN and SUP certifications expire — all others are permanent.
STATIONS = ["BEV", "PREP", "PACK", "CLEAN", "OPEN", "CLOSE", "SUP", "TRAIN"]

# Which stations have expiring certifications, and how long they last (days)
EXPIRY_RULES = {
    "CLEAN": 365,    # annual renewal — mandatory for all staff
    "SUP":   730,    # biennial (every 2 years) — supervisory certification
}

# ── Training type by proficiency level ────────────────────────────────────────
# This maps what the training EVENT is called based on the level achieved.
# Think of it like naming the belt you earned in martial arts.
TRAINING_TYPE_BY_LEVEL = {
    1: "Initial Training",         # just started learning the station
    2: "Competency Check",         # formally assessed, now certified
    3: "Proficiency Assessment",   # exceeded basics, trusted at peak hours
    4: "Trainer Certification",    # certified to teach others
}

# ── Score ranges by proficiency level ─────────────────────────────────────────
# Assessment scores reflect how well the employee performed.
# Higher proficiency = higher score range (they've had more practice).
SCORE_RANGES = {
    1: (0,   0),    # In Progress — not yet assessed, no score
    2: (75, 88),    # Passed competency — solid but still growing
    3: (82, 95),    # Proficient — consistently strong performance
    4: (88, 100),   # Trainer level — near-perfect, trusted to teach
}

# ── Trainer pools per station ─────────────────────────────────────────────────
# For each station, these are the employee IDs who are qualified to train others.
# Rule: must be is_trainer=1 (officially in the Train the Trainer pipeline)
# OR be a Manager (they can always train at any station).
# Managers E001 and E002 are the ultimate fallback for any station.
TRAINER_POOLS = {
    "BEV":   ["E001", "E002", "E003", "E005", "E006", "E008"],
    "PREP":  ["E001", "E002", "E003", "E005", "E006", "E008"],
    "PACK":  ["E001", "E002", "E003", "E005", "E006"],
    "CLEAN": ["E001", "E002", "E003", "E005", "E006", "E008"],
    "OPEN":  ["E001", "E002", "E003", "E005", "E006", "E008"],
    "CLOSE": ["E001", "E002", "E003", "E005"],
    "SUP":   ["E001", "E002", "E003", "E005"],
    "TRAIN": ["E001", "E002"],   # only managers certify new trainers
}

# ── Sample notes by training type ────────────────────────────────────────────
# Realistic short notes a manager or trainer might write in the training log.
NOTES_BY_TYPE = {
    "Initial Training": [
        "Orientation complete. Shadowing next shift.",
        "Strong start. Needs more reps at station.",
        "Good attitude. Speed still developing.",
        "Completed intro module. Assigned buddy trainer.",
        "Quick learner. Recommended for fast-track competency check.",
    ],
    "Competency Check": [
        "Passed assessment. Cleared for independent scheduling.",
        "Met all standards. Recommend continued development.",
        "Solid performance. Ready for solo shifts.",
        "Passed with no critical errors. Certified at Level 2.",
        "Assessment completed. Competency confirmed.",
    ],
    "Proficiency Assessment": [
        "Consistently performs above standard. Reliable at peak hours.",
        "Peak-hour performance confirmed. Certified Level 3.",
        "Speed and accuracy both exceed baseline. Proficient.",
        "Recommended for peak scheduling priority at this station.",
        "Level 3 confirmed. A go-to crew member for this station.",
    ],
    "Trainer Certification": [
        "Completed Train the Trainer module. Cleared to onboard new hires.",
        "Trainer assessment passed. Added to active trainer roster.",
        "Excellent demonstration skills. Level 4 confirmed.",
        "Certified trainer. First assignment: shadow E001 next onboarding.",
        "Exceptional performance throughout trainer program. Certified.",
    ],
    "Annual Renewal": [
        "CLEAN annual renewal completed. Certificate updated.",
        "Mandatory renewal passed. No new gaps identified.",
        "Annual refresher complete. Cert renewed for next 12 months.",
        "Renewal assessment passed on first attempt.",
        "CLEAN recertification done. File updated.",
    ],
}

# ── Date helpers ──────────────────────────────────────────────────────────────
# We generate realistic training dates spread over 2022-2024.
# Longer-tenured employees (managers, supervisors) have older records.
# Newer employees (most PT crew) have more recent training dates.

def random_date(start: date, end: date) -> date:
    """Pick a random date between start and end (inclusive)."""
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

# Approximate "started training" windows by role.
# These are the earliest dates we'd expect to see in their records.
ROLE_START_WINDOWS = {
    "Manager":          (date(2019, 1, 1), date(2021, 6, 30)),
    "Shift Supervisor": (date(2020, 6, 1), date(2022, 6, 30)),
    "Crew - Full Time": (date(2021, 6, 1), date(2023, 3, 31)),
    "Crew - Part Time": (date(2022, 6, 1), date(2023, 12, 31)),
}

END_OF_DATA = date(2024, 12, 31)   # our data year ends here

# ── Helper: pick a trainer for a given station, excluding the trainee ─────────
def pick_trainer(station: str, exclude_emp_id: str) -> str:
    """
    Returns a trainer's emp_id for the given station.
    Excludes the employee being trained (you can't train yourself).
    Falls back to E001 (Maria Santos, Manager) if no one else qualifies.
    """
    pool = [t for t in TRAINER_POOLS[station] if t != exclude_emp_id]
    if not pool:
        return "E001"   # ultimate fallback: head manager always qualifies
    return random.choice(pool)

# ── Generate training records ─────────────────────────────────────────────────
print("Generating training records...")
rows           = []
record_counter = 0

for _, emp in emp_df.iterrows():

    emp_id   = emp["emp_id"]
    role     = emp["role"]

    # Get the date window for when this employee started their training journey
    start_window, end_window = ROLE_START_WINDOWS[role]

    for station in STATIONS:

        # Read this employee's proficiency level at this station
        level = int(emp[f"station_{station}"])

        # Skip stations they have never trained at (Level 0 = not trained)
        if level == 0:
            continue

        # ── Case 1: Level 1 — Initial Training, still in progress ────────────
        # They've started but haven't passed their competency check yet.
        # Think of it as a new hire who started learning the BEV station
        # last month but hasn't been formally assessed.
        if level == 1:
            train_date = random_date(date(2024, 6, 1), date(2024, 11, 30))
            trainer_id = pick_trainer(station, emp_id)

            # Followup date: 4 weeks after training started — scheduled check-in
            followup = train_date + timedelta(weeks=4)

            record_counter += 1
            rows.append({
                "record_id":          f"TR{record_counter:03d}",
                "employee_id":        emp_id,
                "station_code":       station,
                "training_type":      "Initial Training",
                "trainer_id":         trainer_id,
                "training_date":      train_date.strftime("%Y-%m-%d"),
                "score":              None,              # not yet assessed
                "proficiency_level":  1,
                "certification_date": None,              # not yet certified
                "expiry_date":        None,
                "is_expired":         0,
                "followup_date":      followup.strftime("%Y-%m-%d"),
                "followup_done":      0,                 # still pending
                "notes":              random.choice(NOTES_BY_TYPE["Initial Training"]),
                "status":             "In Progress",
            })

        # ── Case 2: Level 2, 3, or 4 — Two records per station ───────────────
        # Record A: the historical Initial Training (completed, older date)
        # Record B: the current certification level (newer date)
        #
        # Think of Record A as the "I started learning" entry in the binder
        # and Record B as the "I passed and got my stamp" entry.
        else:
            # ── Record A: Initial Training (historical) ───────────────────────
            # This happened earlier in their career. Pick a date in their
            # role start window, leaving at least 60 days before the cert.
            init_end   = min(end_window, date(2024, 10, 31))
            init_date  = random_date(start_window, init_end)
            trainer_id = pick_trainer(station, emp_id)

            # Initial training score: set if they passed quickly, else blank
            init_score = int(np.random.randint(65, 80))

            record_counter += 1
            rows.append({
                "record_id":          f"TR{record_counter:03d}",
                "employee_id":        emp_id,
                "station_code":       station,
                "training_type":      "Initial Training",
                "trainer_id":         trainer_id,
                "training_date":      init_date.strftime("%Y-%m-%d"),
                "score":              init_score,
                "proficiency_level":  1,                 # the level AT THAT TIME
                "certification_date": None,
                "expiry_date":        None,
                "is_expired":         0,
                "followup_date":      (init_date + timedelta(weeks=4)).strftime("%Y-%m-%d"),
                "followup_done":      1,                 # completed long ago
                "notes":              random.choice(NOTES_BY_TYPE["Initial Training"]),
                "status":             "Completed",
            })

            # ── Record B: Current certification level ─────────────────────────
            # This happened at least 60 days after initial training —
            # enough time to practise before a formal assessment.
            cert_start = init_date + timedelta(days=60)
            cert_end   = min(END_OF_DATA, cert_start + timedelta(days=365))
            # If the window is too narrow, just add 60 days to start
            if cert_end <= cert_start:
                cert_end = cert_start + timedelta(days=60)
            cert_date  = random_date(cert_start, cert_end)

            # Score based on the level achieved (higher level = higher score)
            lo, hi = SCORE_RANGES[level]
            score  = int(np.random.randint(lo, hi + 1))

            training_type = TRAINING_TYPE_BY_LEVEL[level]
            notes_pool    = NOTES_BY_TYPE[training_type]

            # Expiry logic: CLEAN and SUP certs expire; others are permanent
            if station in EXPIRY_RULES:
                expiry = cert_date + timedelta(days=EXPIRY_RULES[station])
                expired = 1 if expiry < END_OF_DATA else 0
            else:
                expiry  = None
                expired = 0

            # Status: expired certs are flagged for renewal
            if expiry and expired:
                status = "Pending Renewal"
            else:
                status = "Completed"

            # Followup: only needed for expired certs (schedule renewal)
            if expired:
                followup_date = (END_OF_DATA + timedelta(weeks=2)).strftime("%Y-%m-%d")
                followup_done = 0
            else:
                followup_date = None
                followup_done = 1

            record_counter += 1
            rows.append({
                "record_id":          f"TR{record_counter:03d}",
                "employee_id":        emp_id,
                "station_code":       station,
                "training_type":      training_type,
                "trainer_id":         trainer_id,
                "training_date":      cert_date.strftime("%Y-%m-%d"),
                "score":              score,
                "proficiency_level":  level,
                "certification_date": cert_date.strftime("%Y-%m-%d"),
                "expiry_date":        expiry.strftime("%Y-%m-%d") if expiry else None,
                "is_expired":         expired,
                "followup_date":      followup_date,
                "followup_done":      followup_done,
                "notes":              random.choice(notes_pool),
                "status":             status,
            })

# ── Annual CLEAN Renewal records ──────────────────────────────────────────────
# CLEAN certification is mandatory for ALL staff — it's the Food Handler
# safety requirement. Every employee must renew it annually.
# This block generates a 2024 renewal record for each employee,
# separate from the initial certification generated above.
#
# In a real store, this is the record you pull during a health inspection
# to prove every person on the floor is currently food-safe certified.
print("  Adding annual CLEAN renewal records for all staff...")

for _, emp in emp_df.iterrows():
    emp_id    = emp["emp_id"]
    # Renewal date: spread across Jan-Sep 2024 (staggered throughout the year)
    renewal_date = random_date(date(2024, 1, 15), date(2024, 9, 30))
    expiry_date  = renewal_date + timedelta(days=365)
    expired      = 1 if expiry_date < END_OF_DATA else 0
    trainer_id   = pick_trainer("CLEAN", emp_id)

    record_counter += 1
    rows.append({
        "record_id":          f"TR{record_counter:03d}",
        "employee_id":        emp_id,
        "station_code":       "CLEAN",
        "training_type":      "Annual Renewal",
        "trainer_id":         trainer_id,
        "training_date":      renewal_date.strftime("%Y-%m-%d"),
        "score":              int(np.random.randint(80, 100)),
        "proficiency_level":  int(emp["station_CLEAN"]),  # their current level
        "certification_date": renewal_date.strftime("%Y-%m-%d"),
        "expiry_date":        expiry_date.strftime("%Y-%m-%d"),
        "is_expired":         expired,
        "followup_date":      None,
        "followup_done":      1,
        "notes":              random.choice(NOTES_BY_TYPE["Annual Renewal"]),
        "status":             "Completed",
    })

# ── Build DataFrame and save ──────────────────────────────────────────────────
df = pd.DataFrame(rows)

# Sort by employee_id then training_date so the file reads chronologically
df = df.sort_values(["employee_id", "station_code", "training_date"]).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDONE: qsr_training_records.csv saved — {len(df):,} rows | {len(df.columns)} columns")
print()

# ── Verification summary ──────────────────────────────────────────────────────
print("=" * 65)
print("  TRAINING RECORDS SUMMARY")
print("=" * 65)

print(f"\n  Total records     : {len(df):,}")
print(f"  Employees covered : {df['employee_id'].nunique()}")
print(f"  Stations covered  : {df['station_code'].nunique()}")

# Records by training type
print(f"\n  RECORDS BY TRAINING TYPE:")
for ttype, count in df["training_type"].value_counts().items():
    print(f"    {ttype:<28} {count:>4} records")

# Records by status
print(f"\n  RECORDS BY STATUS:")
for status, count in df["status"].value_counts().items():
    print(f"    {status:<20} {count:>4} records")

# Expired certifications — these feed Module 3 Trigger T1
expired = df[df["is_expired"] == 1]
print(f"\n  EXPIRED CERTIFICATIONS ({len(expired)} total) — feeds Trigger T1:")
if expired.empty:
    print("    None expired.")
else:
    exp_detail = expired.merge(
        emp_df[["emp_id", "name"]].rename(columns={"emp_id": "employee_id"}),
        on="employee_id"
    )
    for _, row in exp_detail.iterrows():
        print(f"    {row['employee_id']}  {row['name']:<22}  "
              f"{row['station_code']:<6}  expired: {row['expiry_date']}")

# Certified staff count per station (Level 2+) — Trigger T1 check
print(f"\n  CERTIFIED STAFF PER STATION (Level 2+) — Trigger T1 threshold = 2:")
# Count unique employees with a Completed cert at each station (not expired)
active_certs = df[
    (df["proficiency_level"] >= 2) &
    (df["status"].isin(["Completed"]))
]
for station in STATIONS:
    s_df    = active_certs[active_certs["station_code"] == station]
    unique_certified = s_df["employee_id"].nunique()
    flag = "  <-- WARNING: below minimum!" if unique_certified < 2 else ""
    print(f"    {station:<6} {unique_certified:>3} certified staff{flag}")

# In-progress training (level 1) — upcoming certifications to watch
in_progress = df[df["status"] == "In Progress"]
print(f"\n  IN PROGRESS (Level 1 — not yet certified): {len(in_progress)} records")
if not in_progress.empty:
    ip_detail = in_progress.merge(
        emp_df[["emp_id", "name"]].rename(columns={"emp_id": "employee_id"}),
        on="employee_id"
    )
    for _, row in ip_detail.iterrows():
        print(f"    {row['employee_id']}  {row['name']:<22}  "
              f"{row['station_code']:<6}  followup: {row['followup_date']}")
