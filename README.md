# QSR Operations Platform
### Smart Schedule Optimizer · Training & Development · Hiring Needs Analysis

> **Portfolio Project 1 of 5** — Business Analysis & Product Management  
> Built by Edwin Olaez · Software Development Student · Former QSR General Manager (20+ years)

---

## Table of Contents

1. [The Business Problem](#the-business-problem)
2. [Central Business Theory](#central-business-theory)
3. [Platform Overview — 3 Modules](#platform-overview)
4. [The 6 Morale Factors](#the-6-morale-factors)
5. [Project Architecture](#project-architecture)
6. [The 7 Datasets](#the-7-datasets)
7. [Module 1 — Smart Schedule Optimizer](#module-1--smart-schedule-optimizer)
8. [Module 2 — Training & Development](#module-2--training--development)
9. [Module 3 — Hiring Needs Analysis](#module-3--hiring-needs-analysis)
10. [The ML Model](#the-ml-model)
11. [How to Run](#how-to-run)
12. [Tech Stack](#tech-stack)
13. [Project Status](#project-status)
14. [About the Author](#about-the-author)

---

## The Business Problem

Quick Service Restaurants (QSRs) are operationally complex environments with thin margins, high staff turnover, and intense pressure to deliver consistent customer experiences at high speed. The average QSR General Manager makes scheduling decisions daily with minimal data support — relying on intuition, paper rosters, and spreadsheets that cannot account for demand patterns, staff morale trends, or certification gaps simultaneously.

The result is a predictable cycle:

```
Poor scheduling → staff feel disrespected → morale drops
→ service quality suffers → sales erode → pressure increases
→ scheduling gets worse
```

This platform breaks that cycle by replacing intuition-based scheduling with ML-powered, data-driven operations management.

**Industry context this platform addresses:**
- QSR employee turnover rates average **75–150% annually** — the highest of any industry
- A single unfilled shift costs roughly **3× the hourly wage** in lost productivity and service impact
- Most GM tools today are scheduling apps with no connection to training compliance, morale data, or hiring forecasting

---

## Central Business Theory

The entire platform is built on one hypothesis, grounded in 20+ years of QSR general management experience:

```
Better scheduling  →  higher staff morale  →  better customer experience
                   →  consistent sales growth
```

This is not just a slogan — it is a **measurable, data-driven chain**:

| Link in the Chain | Measured By | Dataset |
|---|---|---|
| Scheduling quality | Hours consistency, days-off respect | `qsr_employee_survey.csv` |
| Staff morale | morale_index (6-factor weighted score) | `qsr_employee_survey.csv` |
| Customer experience | CSAT score, NPS flag, speed of service | `qsr_customer_survey.csv` |
| Sales growth | customer_count, transaction_count, YoY trend | `qsr_sales_data.csv` |

When all three modules are running, a GM can answer: *"Did the schedule I posted last week improve morale? Did that morale improvement show up in CSAT scores? Did the CSAT improvement correlate with sales?"*

That is the full loop. This platform closes it.

---

## Platform Overview

The platform is organized into **3 connected modules**, each solving a distinct operational problem, all sharing the same employee and sales data backbone.

```
┌─────────────────────────────────────────────────────────────┐
│                     QSR OPERATIONS PLATFORM                  │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐│
│  │    MODULE 1      │  │    MODULE 2      │  │ MODULE 3   ││
│  │  Smart Schedule  │  │  Training &      │  │  Hiring    ││
│  │   Optimizer      │  │  Development     │  │  Analysis  ││
│  │                  │  │                  │  │            ││
│  │ ML demand        │  │ Station certs    │  │ 7 triggers ││
│  │ prediction       │  │ Compliance       │  │ Gap detect ││
│  │ 9 hard rules     │  │ Train-the-       │  │ Succession ││
│  │ Weekly schedule  │  │ Trainer pipeline │  │ forecasting││
│  └──────────────────┘  └──────────────────┘  └────────────┘│
│                                                             │
│  Shared foundation: qsr_employees.csv (24 columns)          │
│  Sales, survey, training, compliance, hiring data            │
└─────────────────────────────────────────────────────────────┘
```

---

## The 6 Morale Factors

The morale model is the conceptual heart of the platform. Each factor maps to a specific operational practice — and a specific module feature designed to protect it.

| # | Factor | What It Measures | Module Feature |
|---|---|---|---|
| 1 | **Consistent weekly scheduled hours** | Were hours similar week to week? | Module 1 — Scheduler respects min/max hours |
| 2 | **Consistent communication & training** | Did the employee receive clear info? | Module 2 — Training records and compliance alerts |
| 3 | **Managers on floor during peak hours** | Was a supervisor visible at 12pm/6pm? | Module 1 — SUP rule enforced at every shift |
| 4 | **No stock runouts** | Did the store have everything it needed? | All modules — capacity and staffing correlation |
| 5 | **Respect for personal time / days off** | Were days off and availability honored? | Module 1 — Hard Rules 1, 5, 6, 7 |
| 6 | **Supervisor daily plan with set goals** | Was there a clear plan at shift start? | Module 2 — Training structure and role readiness |

The `morale_index` in the employee survey is a weighted composite of all 6 factors (higher weight on Factors 1 and 5, based on QSR retention research).

---

## Project Architecture

### Repository Structure

```
smart-schedule-optimizer/
│
├── app/
│   └── streamlit_app.py          # Module 1 — Full Streamlit web UI
│
├── data/                         # All 7 generated datasets (gitignored, regenerated by scripts)
│   ├── qsr_sales_data.csv
│   ├── qsr_employees.csv
│   ├── qsr_customer_survey.csv
│   ├── qsr_employee_survey.csv
│   ├── qsr_training_records.csv
│   ├── qsr_compliance_training.csv
│   └── qsr_hiring_analysis.csv
│
├── models/                       # Trained ML models + lookup tables (gitignored)
│   ├── staff_model.pkl           # Stage 2 baseline RandomForest
│   ├── staff_model_enhanced.pkl  # Stage 2b enhanced RandomForest (active)
│   ├── ly_agg_2023.csv           # Last-year hourly aggregates for prediction
│   └── weekly_hourly_2024.csv    # Trend ratio lookup table
│
├── reports/                      # Model summaries + schedule outputs (gitignored)
│   ├── model_summary.txt
│   ├── enhanced_model_summary.txt
│   ├── feature_importance.csv
│   ├── feature_importance_enhanced.csv
│   └── schedule_week_YYYY-MM-DD.txt
│
├── scripts/
│   │
│   │   ── DATA GENERATION ──────────────────────────────────────────
│   ├── stage1_generate_data.py              # QSR sales data (2023–2024)
│   ├── stage1b_generate_employees.py        # Employee roster (12 cols)
│   ├── stage1c_update_employees_stations.py # Adds 12 station/career cols
│   ├── stage1d_customer_survey_data.py      # Customer satisfaction surveys
│   ├── stage1e_employee_survey_data.py      # Employee morale surveys
│   ├── stage2_training_records.py           # Station training history
│   ├── stage2b_compliance_training.py       # Regulatory compliance certs
│   ├── stage3_hiring_analysis.py            # Hiring trigger analysis
│   │
│   │   ── ML MODEL ─────────────────────────────────────────────────
│   ├── stage2_train_model.py                # Baseline RandomForest
│   ├── stage2b_enhanced_model.py            # Enhanced model with LY context
│   │
│   │   ── OPTIMIZER ────────────────────────────────────────────────
│   └── stage3_optimizer.py                  # Schedule generator (9 hard rules)
│
└── requirements.txt
```

---

## The 7 Datasets

All datasets are **synthetically generated** from scripts in the `/scripts` folder. They are designed to be realistic for an Alberta, Canada QSR location. All data is regenerated by running the scripts in order.

| # | File | Rows | Cols | Module | Description |
|---|---|---|---|---|---|
| 1 | `qsr_sales_data.csv` | 13,158 | 13 | 1 | Hourly sales data for 2023–2024 (731 days × 18 hours). Includes customer count, transactions, weather, holidays, promotions, staff needed. |
| 2 | `qsr_employees.csv` | 15 | 24 | All | Employee roster with scheduling constraints, station proficiency scores (0–4 scale), career path, and trainer status. |
| 3 | `qsr_customer_survey.csv` | 1,830 | 16 | 1 | ~5 surveys/day for 2024. Ratings for speed, friendliness, accuracy, cleanliness, product availability. Includes CSAT score and NPS flag. |
| 4 | `qsr_employee_survey.csv` | 780 | 17 | 1 | Weekly morale survey for all 15 employees across 52 weeks. Tracks the 6 morale factors, morale_index, and at_risk_flag for Trigger T4. |
| 5 | `qsr_training_records.csv` | 200 | 15 | 2 | Full training history: initial training, competency checks, proficiency assessments, and trainer certifications per employee per station. |
| 6 | `qsr_compliance_training.csv` | 59 | 9 | 2 | Regulatory compliance certifications (Food Safety, WHMIS, First Aid, OH&S, etc.) with expiry dates, days until expiry, and renewal status. |
| 7 | `qsr_hiring_analysis.csv` | 22 | 15 | 3 | Output of the 7 hiring trigger checks — logged gap events with severity, recommended action, hours gap, and estimated headcount. |

---

## Module 1 — Smart Schedule Optimizer

### What It Does

Takes a target week as input and produces a fully-staffed, rule-compliant weekly schedule. It predicts how many staff are needed per hour using machine learning, builds a shift plan from those predictions, and assigns real employees to shifts while enforcing all 9 hard rules simultaneously.

### The 9 Hard Scheduling Rules

| Rule | Description |
|---|---|
| 1 | Never schedule an employee on their designated day(s) off |
| 2 | Every shift must have at least 1 Supervisor or Manager |
| 3 | Never exceed an employee's maximum weekly hours |
| 4 | Minimum store staffing of 2 people at all times |
| 5 | Always respect each employee's availability window (start/end hour) |
| 6 | No clopening — minimum 10 hours between a close shift and the next open shift |
| 7 | Maximum 5 working days per employee per week |
| 8 | Schedule is locked once posted — no edits without approval |
| 9 | Any schedule change requires the affected employee's approval |

### The 8 Station Codes

| Code | Station | Key Minimum Rule |
|---|---|---|
| BEV | Beverage / Drinks | Min 1 BEV-certified per shift during peak hours |
| PREP | Food Prep / Assembly | Min 2 PREP-certified at all times |
| PACK | Delivery / Order Packing | Min 1 PACK; 2 during dinner peak |
| CLEAN | Cleaning & Sanitation | ALL employees must hold CLEAN cert (mandatory) |
| OPEN | Opening Duties | Min 1 OPEN-certified each morning |
| CLOSE | Closing Duties | Min 1 CLOSE-certified each closing shift |
| SUP | Supervisor / Shift Lead | Min 1 SUP-certified per shift (Hard Rule 2) |
| TRAIN | Team Member Trainer | Qualifies employee to onboard new hires |

### Station Proficiency Scale

```
0 = Not Trained   — never worked this station, never scheduled here
1 = Learning      — in training, must be supervised, never scheduled alone
2 = Competent     — certified, can work this station independently
3 = Proficient    — preferred for peak hours, trusted under pressure
4 = Trainer       — can train others, earns the TRAIN certification flag
```

### The 6 Shift Types

| Shift | Hours | Duration |
|---|---|---|
| OPEN | 06:00 – 14:00 | 8h |
| MID | 10:00 – 18:00 | 8h |
| CLOSE | 15:00 – 23:00 | 8h |
| PT_AM | 06:00 – 12:00 | 6h |
| PT_PM | 12:00 – 18:00 | 6h |
| PT_EVE | 17:00 – 23:00 | 6h |

### The Employee Roster

| ID | Name | Role | Availability | Day(s) Off | BEV/PREP/PACK/CLEAN/OPEN/CLOSE/SUP/TRAIN |
|---|---|---|---|---|---|
| E001 | Maria Santos | Manager | 06–23 | Sun | 3/3/3/4/4/4/4/4 ★ Trainer |
| E002 | James Reyes | Manager | 06–23 | Sat | 3/3/3/4/4/4/4/4 ★ Trainer |
| E003 | Ana Dela Cruz | Shift Supervisor | 06–21 | Fri | 3/3/2/3/3/2/3/3 ★ Trainer |
| E004 | Kevin Park | Shift Supervisor | 10–23 | Mon | 2/3/3/3/1/4/3/2 |
| E005 | Priya Nair | Shift Supervisor | 08–22 | Wed | 3/2/2/3/2/3/3/3 ★ Trainer |
| E006 | Carlos Mendez | Crew – Full Time | 06–22 | Tue | 3/4/3/3/3/2/0/3 ★ Trainer |
| E007 | Sophie Tremblay | Crew – Full Time | 07–23 | Thu | 2/3/2/3/1/3/0/0 |
| E008 | Diego Rivera | Crew – Full Time | 06–21 | Sun | 4/3/2/3/3/1/0/4 ★ Trainer |
| E009 | Fatima Al-Hassan | Crew – Full Time | 09–23 | Sat | 3/2/3/2/0/3/0/2 |
| E010 | Liam Johnson | Crew – Part Time | 14–23 | Mon, Wed | 2/2/3/2/0/2/0/0 |
| E011 | Aisha Mohammed | Crew – Part Time | 06–15 | Tue, Thu | 3/2/1/3/3/0/0/0 |
| E012 | Emma Leblanc | Crew – Part Time | 10–19 | Mon, Fri | 1/2/2/2/1/0/0/0 |
| E013 | Noah Kim | Crew – Part Time | 15–23 | Wed, Sun | 2/2/2/2/0/2/0/0 |
| E014 | Isabelle Okonkwo | Crew – Part Time | 06–18 | Sat | 2/3/2/3/2/0/0/0 |
| E015 | Marco Ferreira | Crew – Part Time | 11–23 | Sun, Mon | 1/2/2/2/0/2/0/0 |

### Streamlit Web App (Stage 4)

The optimizer is wrapped in an interactive Streamlit app with:

- **Monthly calendar UI** — click any week to select the scheduling period
- **Alberta holiday detection** — holidays automatically flagged on the calendar (long weekends, Family Day, Heritage Day, etc.)
- **Per-day overrides** — adjust weather, promotions, and event type for each day individually
- **Event presets** — 8 preset traffic multipliers (festivals, school breaks, construction, competitor closures, etc.) plus a custom multiplier input
- **Reliability scoring** — each day shows a confidence percentage (HIGH/MEDIUM/LOW) based on how far in advance you're scheduling and how stable last-year demand was for that week
- **Schedule output** — full weekly schedule with employee names, roles, shifts, hours, and peak staff projections

---

## Module 2 — Training & Development

### What It Does

Tracks every training event for every employee at every station. Surfaces who needs to renew a certification before it expires, who is currently in training and needs a followup, and who is progressing through the Train the Trainer pipeline.

### The Train the Trainer Pipeline (5 Stages)

```
Stage 1: Identified    → Level 3 at 2+ stations, nominated by manager
Stage 2: Shadowing     → Observes 3 training sessions, provides feedback
Stage 3: Co-Delivery   → Delivers 2 sessions with Level 4 supervisor present
Stage 4: Certified     → Passes trainer assessment, earns Level 4, joins trainer roster
Stage 5: Mentoring     → Active 6+ months, feeds the leadership pipeline
```

**Active trainers in this store:** Maria Santos, James Reyes, Ana Dela Cruz, Priya Nair, Carlos Mendez, Diego Rivera

### Compliance Certification Types

| Certification | Issuing Body | Expiry | Who Needs It |
|---|---|---|---|
| Food Safety Level 1 | Alberta Health Services | 5 years | All staff |
| WHMIS 2015 | Government of Canada / CCOHS | 1 year | All staff |
| Allergen Awareness | QSR Internal (Company Policy) | 1 year | All staff |
| First Aid Level 1 | Canadian Red Cross | 3 years | Managers + Supervisors |
| OH&S Fundamentals | Government of Alberta | 3 years | Managers, Supervisors, Senior FT |
| Fire Safety Awareness | City Fire Prevention Office | 1 year | Managers only |

---

## Module 3 — Hiring Needs Analysis

### What It Does

Reads all 7 datasets and runs 7 programmatic trigger checks every week. When a trigger fires, it logs a record with the gap description, severity, recommended action, hours gap per week, and estimated headcount to hire.

### The 7 Hiring Triggers

| Code | Trigger | Severity | Data Source |
|---|---|---|---|
| T1 | Station has fewer than 2 certified staff (Level 2+) | High | `qsr_training_records.csv` |
| T2 | Any regular shift window has only 1 available supervisor | High | `qsr_employees.csv` |
| T3 | ML model predicts volume >20% above capacity for 2+ consecutive weeks | Medium | `qsr_sales_data.csv` |
| T4 | Employee morale_index <3.0 for 3+ consecutive weeks (flight risk) | High | `qsr_employee_survey.csv` |
| T5 | Any role or certification has only 1 current holder | Medium | `qsr_employees.csv` + compliance |
| T6 | Weekly hours demand exceeds total team capacity | Medium | `qsr_sales_data.csv` + `qsr_employees.csv` |
| T7 | Employee is promotion-ready (bench_priority=1) — creates vacancy below | Low | `qsr_employees.csv` |

### Key Findings from the 2024 Analysis

The hiring analysis detected **22 trigger events** across 2024:

- **T1 (Critical):** SUP station has only 1 employee with an active (non-expired) certification. One absence = no qualified shift lead.
- **T4 (Resolved):** Emma Leblanc and Marco Ferreira both hit the morale floor (index < 3.0 for 3+ consecutive weeks). Both recovered after intervention.
- **T5 (Open):** Only 2 Managers in the store. One departure creates a critical role gap with no internal cover.
- **T6 (Systemic):** Average weekly demand (427 hrs) exceeds 90% of team capacity (311 hrs) every week of 2024. The store needs 2 additional part-time crew members to close a 116-hr/week structural labor gap.
- **T7 (In Progress):** Ana Dela Cruz (Supervisor → Manager) and Carlos Mendez (FT → Supervisor) are both promotion-ready at bench_priority=1 — creating two downstream vacancies that need proactive backfill recruiting.

---

## The ML Model

### Why Machine Learning for Scheduling?

Traditional scheduling relies on last week's schedule plus the GM's memory of what was busy. Machine learning replaces that with a model trained on 13,158 rows of hourly sales history — capturing the complex interaction of time of day, day of week, weather, holidays, promotions, and last-year demand simultaneously.

### Model Evolution

#### Stage 2 — Baseline Model

| Parameter | Value |
|---|---|
| Algorithm | RandomForestRegressor |
| Target | `staff_needed` (rounded integer, min 2) |
| Features | 6 (hour, day, weekend, holiday, promotion, weather) |
| Training rows | 5,270 |
| MAE | 0.237 (avg off by 0.24 staff members) |
| R² | 0.8976 |

#### Stage 2b — Enhanced Model (Active)

| Parameter | Value |
|---|---|
| Algorithm | RandomForestRegressor (300 trees, max depth 12) |
| Target | `staff_needed` (rounded integer, min 2) |
| Features | 11 (adds LY context, trend ratio, LY same-week data) |
| Training year | 2024 with 2023 as reference |
| MAE | **0.218** (avg off by 0.22 staff members) |
| R² | **0.9117** |

#### Feature Importance (Enhanced Model)

| Feature | Importance | What It Represents |
|---|---|---|
| `ly_context_cust` | 57.6% | Avg customers, same ISO week ±3 weeks in last year |
| `ly_context_txn` | 25.1% | Avg transactions, same context window |
| `weather_num` | 4.5% | Weather condition (sunny/cloudy/rainy/snowy) |
| `day_of_week_num` | 2.8% | Day of the week (0=Monday) |
| `is_weekend` | 2.7% | Whether it's Saturday or Sunday |
| `ly_same_week_cust` | 2.5% | Avg customers, exact same week last year |
| Others | 4.8% | Holiday, promotion, hour, trend ratio, LY same week txn |

**Key insight:** Last-year context (LY) accounts for **82.7% of predictive power**. Time of day (`hour`) alone — which drives most rule-based schedulers — accounts for just 1.1%. This validates using a data-driven approach over intuition.

### Reliability Scoring

Because predictions become less reliable the further in advance you schedule, the model includes a decay-based reliability score:

```
base_score  = 1.0 - (LY_coefficient_of_variation × 1.5)
base_score  = clamped to [0.50, 0.95]
reliability = base_score × 0.97^(weeks_ahead)

HIGH   ≥ 85%   — schedule with confidence
MEDIUM ≥ 72%   — schedule with normal review
LOW    < 72%   — treat as estimate, review before posting
```

---

## How to Run

### Prerequisites

- Python 3.11+
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/smart-schedule-optimizer.git
cd smart-schedule-optimizer

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Generate All Data (Run in Order)

```bash
# Module 1 — Sales data and employee roster
python scripts/stage1_generate_data.py         # generates qsr_sales_data.csv
python scripts/stage1b_generate_employees.py   # generates qsr_employees.csv (12 cols)
python scripts/stage1c_update_employees_stations.py  # upgrades to 24 cols

# Module 1 — Survey data
python scripts/stage1d_customer_survey_data.py # generates qsr_customer_survey.csv
python scripts/stage1e_employee_survey_data.py # generates qsr_employee_survey.csv

# ML Models
python scripts/stage2_train_model.py           # trains baseline model
python scripts/stage2b_enhanced_model.py       # trains enhanced model (use this one)

# Module 2 — Training records
python scripts/stage2_training_records.py      # generates qsr_training_records.csv
python scripts/stage2b_compliance_training.py  # generates qsr_compliance_training.csv

# Module 3 — Hiring analysis
python scripts/stage3_hiring_analysis.py       # generates qsr_hiring_analysis.csv

# Generate a schedule (optional — the app does this interactively)
python scripts/stage3_optimizer.py             # generates schedule for week of 2024-01-08
python scripts/stage3_optimizer.py 2024-06-03  # any Monday date
python scripts/stage3_optimizer.py 2024-06-03 1  # 1 week ahead (higher reliability)
```

### Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| pandas | 2.3.3 | Data manipulation and analysis |
| numpy | 2.4.3 | Numerical computation, random generation |
| scikit-learn | 1.8.0 | RandomForestRegressor ML model |
| matplotlib | 3.10.8 | Visualization (model diagnostics) |
| seaborn | 0.13.2 | Statistical visualization |
| streamlit | 1.55.0 | Interactive web application |

---

## Project Status

### Module 1 — Smart Schedule Optimizer

| Component | Status |
|---|---|
| Sales data generation (2 years, 13,158 rows) | ✅ Complete |
| Employee roster (24 columns, station proficiency) | ✅ Complete |
| Baseline ML model (R² = 0.8976) | ✅ Complete |
| Enhanced ML model with LY context (R² = 0.9117) | ✅ Complete |
| Schedule optimizer — 9 hard rules | ✅ Complete |
| Reliability scoring | ✅ Complete |
| Customer survey data | ✅ Complete |
| Employee morale survey data | ✅ Complete |
| Streamlit app — calendar UI, per-day overrides, event presets | ✅ Complete |
| Station coverage validation in optimizer | 🔲 Planned |
| Schedule lock + employee approval workflow (Rules 8 & 9) | 🔲 Planned |

### Module 2 — Training & Development

| Component | Status |
|---|---|
| Station training records (200 rows, 15 cols) | ✅ Complete |
| Compliance certification records (59 rows, 9 cols) | ✅ Complete |
| Training dashboard (Streamlit) | 🔲 Planned |
| Certification expiry alerts | 🔲 Planned |
| Train the Trainer progress tracker | 🔲 Planned |

### Module 3 — Hiring Needs Analysis

| Component | Status |
|---|---|
| Hiring trigger engine (T1–T7) | ✅ Complete |
| Hiring analysis dataset (22 events, 15 cols) | ✅ Complete |
| Hiring dashboard (Streamlit) | 🔲 Planned |
| Succession planning view | 🔲 Planned |
| Proactive alert system | 🔲 Planned |

---

## About the Author

**Edwin Olaez**  
Software Development Student · Former QSR General Manager

20+ years as a General Manager across five major QSR brands:

| Brand | Tenure |
|---|---|
| Starbucks | 5 years |
| McDonald's | 9 years |
| Tim Hortons | 6 years |
| Seattle Best Coffee | 4 years |
| Shakey's Pizza | 3 years |

**Education:** B.S. Electronics & Communications Engineering  
**Career Goal:** Product Management and Business Analysis in tech

This project exists at the intersection of my two worlds — the operational realities of running a restaurant and the analytical tools of software development. Every scheduling rule, every morale factor, every hiring trigger in this platform reflects a real decision I made, a mistake I witnessed, or a system I wished existed when I was managing a team of 40+ people at peak hour.

This is **Portfolio Project 1 of 5** in my BA/PM student portfolio. The other four projects will apply similar data-driven thinking to different business domains.

---

*Built with Python · scikit-learn · Streamlit · pandas*  
*Alberta, Canada QSR context · 2024 data year*
