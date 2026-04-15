"""
Stage 1c — Add Station Proficiency & Career Data to Employee Roster
====================================================================
WHY THIS MATTERS (Business Reason):
  In a real restaurant, not every crew member can work every station.
  You can't put a brand-new hire on the beverage bar during a lunch rush,
  and you can't run a closing shift without someone CLOSE-certified.

  Right now our employee file only tracks WHEN people can work.
  This script adds WHAT they can do and HOW WELL they can do it.
  That unlocks smarter scheduling: instead of just filling a shift,
  the optimizer can check "do we have a BEV-certified person on the floor?"

WHAT THIS SCRIPT DOES:
  1. Loads the existing qsr_employees.csv (12 columns, written by Stage 1b)
  2. Adds 12 new columns:
       - 8 station proficiency scores (0-4 scale)
       - is_trainer flag
       - next_role (career path target)
       - promotion_ready flag
       - bench_priority score
  3. Overwrites qsr_employees.csv with the full 24-column version

THE 8 STATIONS:
  BEV   — Beverage/Drinks
  PREP  — Food Prep/Assembly
  PACK  — Delivery/Order Packing
  CLEAN — Cleaning & Sanitation (all staff must hold this)
  OPEN  — Opening Duties
  CLOSE — Closing Duties
  SUP   — Supervisor/Shift Lead
  TRAIN — Team Member Trainer (can onboard new hires)

PROFICIENCY SCALE (think of it like your crew development board):
  0 = Not Trained    — never worked this station
  1 = Learning       — in training, needs supervision, never scheduled alone
  2 = Competent      — certified, can work independently
  3 = Proficient     — your go-to person, preferred for peak hours
  4 = Trainer        — so good they can teach others

Outputs:
  data/qsr_employees.csv  (overwrites, now 24 columns)
"""

import pandas as pd  # pandas = our spreadsheet tool in Python
import os            # os = helps us build file paths that work on any computer

# ── File path setup ──────────────────────────────────────────────────────────
# Think of this as navigating from the scripts folder up one level to the
# project root, then into the data folder — same as clicking around in Finder.
BASE        = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EMP_PATH    = os.path.join(BASE, "data", "qsr_employees.csv")

# ── Load the existing employee file ─────────────────────────────────────────
# We're picking up where Stage 1b left off — don't re-create, just add to it.
print("Loading existing employee roster...")
df = pd.read_csv(EMP_PATH)
print(f"  {len(df)} employees loaded | {len(df.columns)} columns currently")

# ── Station proficiency data ─────────────────────────────────────────────────
# Each row below matches one employee (E001–E015).
# Format: emp_id, then scores for BEV, PREP, PACK, CLEAN, OPEN, CLOSE, SUP, TRAIN
#
# Think of this as your paper crew development tracker — the card on the back
# room wall that shows each person's skill level at every station.
# Scores come from the Platform Design document.

station_data = [
    # emp_id  BEV  PREP  PACK  CLEAN  OPEN  CLOSE  SUP  TRAIN
    ("E001",   3,    3,    3,    4,     4,    4,     4,   4),   # Maria Santos     — Manager, full trainer
    ("E002",   3,    3,    3,    4,     4,    4,     4,   4),   # James Reyes      — Manager, full trainer
    ("E003",   3,    3,    2,    3,     3,    2,     3,   3),   # Ana Dela Cruz    — Supervisor, trainer
    ("E004",   2,    3,    3,    3,     1,    4,     3,   2),   # Kevin Park       — Supervisor, OPEN still learning
    ("E005",   3,    2,    2,    3,     2,    3,     3,   3),   # Priya Nair       — Supervisor, trainer
    ("E006",   3,    4,    3,    3,     3,    2,     0,   3),   # Carlos Mendez    — FT, PREP expert, no SUP yet
    ("E007",   2,    3,    2,    3,     1,    3,     0,   0),   # Sophie Tremblay  — FT, solid crew
    ("E008",   4,    3,    2,    3,     3,    1,     0,   4),   # Diego Rivera     — FT, BEV expert + trainer
    ("E009",   3,    2,    3,    2,     0,    3,     0,   2),   # Fatima Al-Hassan — FT, no OPEN availability
    ("E010",   2,    2,    3,    2,     0,    2,     0,   0),   # Liam Johnson     — PT evening, no OPEN
    ("E011",   3,    2,    1,    3,     3,    0,     0,   0),   # Aisha Mohammed   — PT morning, no CLOSE availability
    ("E012",   1,    2,    2,    2,     1,    0,     0,   0),   # Emma Leblanc     — PT midday, still developing
    ("E013",   2,    2,    2,    2,     0,    2,     0,   0),   # Noah Kim         — PT evening, solid basics
    ("E014",   2,    3,    2,    3,     2,    0,     0,   0),   # Isabelle Okonkwo — PT morning, no CLOSE availability
    ("E015",   1,    2,    2,    2,     0,    2,     0,   0),   # Marco Ferreira   — PT, still building skills
]

# ── Convert station data into a DataFrame we can merge ──────────────────────
# A DataFrame is like a mini spreadsheet. We're building one just for the
# station scores so we can stitch it onto the existing employee table.
station_cols = ["emp_id",
                "station_BEV", "station_PREP", "station_PACK", "station_CLEAN",
                "station_OPEN", "station_CLOSE", "station_SUP", "station_TRAIN"]

station_df = pd.DataFrame(station_data, columns=station_cols)

# ── Merge station scores onto the main employee table ───────────────────────
# "merge on emp_id" = "match each row by employee ID" — like a VLOOKUP in Excel.
# how="left" means: keep all existing employees even if a station row is missing.
df = df.merge(station_df, on="emp_id", how="left")

print(f"  Station proficiency columns added.")

# ── Career path data ─────────────────────────────────────────────────────────
# This is the succession planning layer — the "who's next?" board.
#
# next_role        = what role this employee is being developed toward
# promotion_ready  = 1 if they're ready NOW, 0 if still developing
# bench_priority   = 1 = highest succession priority (will create a vacancy below them)
#                    2 = secondary pipeline candidate
#                    0 = not yet in active succession planning
#
# This data feeds Module 3 (Hiring Needs Analysis) — specifically Trigger T7:
# "Employee promotion_ready + bench_priority=1 creates vacancy below."
# Knowing who's about to be promoted lets us hire their backfill proactively.

career_data = [
    # emp_id  next_role                  promotion_ready  bench_priority
    ("E001",  "Area Manager",            0,               0),  # Maria    — already at top of store level
    ("E002",  "Area Manager",            0,               0),  # James    — already at top of store level
    ("E003",  "Manager",                 1,               1),  # Ana      — ready to step up, top priority
    ("E004",  "Manager",                 0,               0),  # Kevin    — still developing (OPEN=1 gap)
    ("E005",  "Manager",                 1,               2),  # Priya    — ready, secondary pipeline
    ("E006",  "Shift Supervisor",        1,               1),  # Carlos   — ready for sup role, top FT pick
    ("E007",  "Shift Supervisor",        0,               0),  # Sophie   — developing toward sup
    ("E008",  "Shift Supervisor",        1,               2),  # Diego    — ready, secondary pipeline
    ("E009",  "Shift Supervisor",        0,               0),  # Fatima   — developing, OPEN gap
    ("E010",  "Crew - Full Time",        0,               0),  # Liam     — building toward full time
    ("E011",  "Crew - Full Time",        0,               0),  # Aisha    — building toward full time
    ("E012",  "Crew - Full Time",        0,               0),  # Emma     — building toward full time
    ("E013",  "Crew - Full Time",        0,               0),  # Noah     — building toward full time
    ("E014",  "Crew - Full Time",        0,               0),  # Isabelle — building toward full time
    ("E015",  "Crew - Full Time",        0,               0),  # Marco    — building toward full time
]

career_cols = ["emp_id", "next_role", "promotion_ready", "bench_priority"]
career_df   = pd.DataFrame(career_data, columns=career_cols)

# Merge career path data the same way we merged stations
df = df.merge(career_df, on="emp_id", how="left")

# ── is_trainer flag ──────────────────────────────────────────────────────────
# A trainer is someone whose station_TRAIN score is 3 or 4 AND who has been
# officially nominated (marked in the original design doc).
# Think of it as the difference between "could train someone" vs
# "is an active part of the Train the Trainer pipeline."
#
# Trainers: E001, E002, E003, E005, E006, E008 (marked with * in the brief)
TRAINER_IDS = {"E001", "E002", "E003", "E005", "E006", "E008"}

# For each employee, check if their ID is in our trainer set — gives 1 or 0
df["is_trainer"] = df["emp_id"].apply(lambda eid: 1 if eid in TRAINER_IDS else 0)

# ── Reorder columns to match the 24-column spec ──────────────────────────────
# The order matters for readability — keep the original 12 first,
# then add all the new ones in a logical group.
FINAL_COLUMN_ORDER = [
    # Original 12 columns (from Stage 1b)
    "emp_id", "name", "role",
    "min_hours", "max_hours", "max_days_per_week",
    "days_off", "availability_start", "availability_end",
    "preferred_shift", "min_rest_hours", "hourly_rate",
    # 8 station proficiency scores (new)
    "station_BEV", "station_PREP", "station_PACK", "station_CLEAN",
    "station_OPEN", "station_CLOSE", "station_SUP", "station_TRAIN",
    # Trainer flag (new)
    "is_trainer",
    # Career path columns (new)
    "next_role", "promotion_ready", "bench_priority",
]

df = df[FINAL_COLUMN_ORDER]  # reorder — like rearranging columns in Excel

# ── Save back to the same file ────────────────────────────────────────────────
# We overwrite qsr_employees.csv. The old 12-column version is replaced
# by this 24-column version. All downstream scripts (optimizer, app) still
# read this same file — they'll just have more columns available now.
df.to_csv(EMP_PATH, index=False)  # index=False means don't write row numbers

print(f"  Career path + trainer flag columns added.")
print(f"\nDONE: qsr_employees.csv updated — {len(df)} employees | {len(df.columns)} columns")
print()

# ── Print a verification summary ─────────────────────────────────────────────
# This is your "sanity check" — like reading back the order on a ticket
# to make sure you didn't miss anything before sending it to the kitchen.
print("=" * 70)
print("  STATION PROFICIENCY SUMMARY")
print("=" * 70)

station_display_cols = [
    "emp_id", "name",
    "station_BEV", "station_PREP", "station_PACK", "station_CLEAN",
    "station_OPEN", "station_CLOSE", "station_SUP", "station_TRAIN",
    "is_trainer"
]
print(df[station_display_cols].to_string(index=False))

print()
print("=" * 70)
print("  CAREER PATH SUMMARY")
print("=" * 70)

career_display_cols = [
    "emp_id", "name", "role", "next_role", "promotion_ready", "bench_priority"
]
print(df[career_display_cols].to_string(index=False))

print()
print("=" * 70)
print("  QUICK CHECKS (minimum staffing requirements)")
print("=" * 70)

# Check: all staff must hold CLEAN cert (Level 2+) — Hard Rule from brief
clean_certified = df[df["station_CLEAN"] >= 2]
print(f"  CLEAN certified (Level 2+): {len(clean_certified)}/15 employees")
if len(clean_certified) < 15:
    not_clean = df[df["station_CLEAN"] < 2][["emp_id","name","station_CLEAN"]]
    print(f"  WARNING — not CLEAN certified:")
    print(not_clean.to_string(index=False))

# Check: how many can open (OPEN = Level 2+)
open_certified = df[df["station_OPEN"] >= 2]
print(f"  OPEN certified  (Level 2+): {len(open_certified)} employees")

# Check: how many can close (CLOSE = Level 2+)
close_certified = df[df["station_CLOSE"] >= 2]
print(f"  CLOSE certified (Level 2+): {len(close_certified)} employees")

# Check: how many are SUP certified (Level 2+ = can run a shift)
sup_certified = df[df["station_SUP"] >= 2]
print(f"  SUP certified   (Level 2+): {len(sup_certified)} employees")

# Check: how many are active trainers
trainers = df[df["is_trainer"] == 1]
print(f"  Active trainers           : {len(trainers)} employees")

# Check: who is promotion ready right now
ready = df[df["promotion_ready"] == 1][["name","role","next_role","bench_priority"]]
print(f"  Promotion ready           : {len(ready)} employees")
if len(ready) > 0:
    print(ready.to_string(index=False))
