"""
Module 2 — Training & Compliance Dashboard
Run with: streamlit run app/module2_training.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Training & Compliance — QSR Ops",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
.kpi-expired   { background:#f8d7da; border-radius:8px; padding:12px; text-align:center; }
.kpi-soon      { background:#fff3cd; border-radius:8px; padding:12px; text-align:center; }
.kpi-current   { background:#d4edda; border-radius:8px; padding:12px; text-align:center; }
.kpi-neutral   { background:#e2e8f0; border-radius:8px; padding:12px; text-align:center; }
.stage-badge   { display:inline-block; padding:3px 10px; border-radius:12px;
                 font-size:0.82rem; font-weight:600; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    emp_df        = pd.read_csv(os.path.join(ROOT, "data", "qsr_employees.csv"))
    training_df   = pd.read_csv(os.path.join(ROOT, "data", "qsr_training_records.csv"))
    compliance_df = pd.read_csv(os.path.join(ROOT, "data", "qsr_compliance_training.csv"))
    return emp_df, training_df, compliance_df

emp_df, training_df, compliance_df = load_data()

# Merge employee names into both datasets for display
emp_names = emp_df[["emp_id", "name", "role"]].rename(columns={"emp_id": "employee_id"})
training_df   = training_df.merge(emp_names,   on="employee_id", how="left")
compliance_df = compliance_df.merge(emp_names, on="employee_id", how="left")

# Snapshot date — all data is "as of" Dec 31 2024
SNAPSHOT = date(2024, 12, 31)

STATIONS       = ["BEV", "PREP", "PACK", "CLEAN", "OPEN", "CLOSE", "SUP", "TRAIN"]
STATION_LABELS = {
    "BEV":   "Beverages",
    "PREP":  "Food Prep",
    "PACK":  "Packaging",
    "CLEAN": "Cleaning",
    "OPEN":  "Opening",
    "CLOSE": "Closing",
    "SUP":   "Supervisor",
    "TRAIN": "Train-the-Trainer",
}
PROFICIENCY_LABELS = {0: "Not Trained", 1: "Basic", 2: "Competent", 3: "Proficient", 4: "Trainer"}
PROFICIENCY_COLORS = {
    0: "#dee2e6",   # gray  — not trained
    1: "#fff3cd",   # amber — basic
    2: "#ffeaa7",   # yellow — competent
    3: "#a8d8a8",   # light green — proficient
    4: "#2d6a4f",   # dark green — trainer
}

STAGE_COLORS = {
    "Mentoring":  ("#2d6a4f", "#ffffff"),
    "Lead":       ("#52b788", "#000000"),
    "Co-Train":   ("#95d5b2", "#000000"),
    "Shadow":     ("#fff3cd", "#856404"),
    "Identified": ("#e2e8f0", "#495057"),
}

STATUS_COLORS = {
    "Expired":       "background-color:#f8d7da;color:#721c24;font-weight:700",
    "Expiring Soon": "background-color:#fff3cd;color:#856404;font-weight:600",
    "Current":       "background-color:#d4edda;color:#155724",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Training & Compliance Dashboard")
st.caption(
    "Station proficiency · compliance certifications · train-the-trainer pipeline  ·  "
    "Alberta QSR regulatory requirements  ·  Snapshot: Dec 31, 2024"
)
st.divider()

# ── Top KPI metrics ───────────────────────────────────────────────────────────
total_certs   = len(compliance_df)
n_expired     = (compliance_df["renewal_status"] == "Expired").sum()
n_soon        = (compliance_df["renewal_status"] == "Expiring Soon").sum()
n_current     = (compliance_df["renewal_status"] == "Current").sum()
comp_rate     = round(n_current / total_certs * 100)
n_trainers    = int(emp_df["is_trainer"].sum())
n_cert_types  = compliance_df["certification_name"].nunique()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Cert Records",     total_certs)
c2.metric("✅ Current",        n_current)
c3.metric("⚠️ Expiring Soon",  n_soon,   delta=f"-{n_soon} need renewal",  delta_color="inverse")
c4.metric("🚨 Expired",        n_expired, delta=f"-{n_expired} action req", delta_color="inverse")
c5.metric("Compliance Rate",  f"{comp_rate}%")
c6.metric("Active Trainers",  n_trainers)

st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")

    emp_options = ["All Employees"] + sorted(emp_df["name"].tolist())
    sel_emp     = st.selectbox("Employee", emp_options)

    cert_options = ["All Certifications"] + sorted(compliance_df["certification_name"].unique().tolist())
    sel_cert     = st.selectbox("Certification Type", cert_options)

    status_options = st.multiselect(
        "Compliance Status",
        ["Expired", "Expiring Soon", "Current"],
        default=["Expired", "Expiring Soon", "Current"],
    )

    station_options = ["All Stations"] + STATIONS
    sel_station     = st.selectbox("Station (Training)", station_options)

    st.divider()
    st.caption("**Data snapshot:** Dec 31, 2024")
    st.caption(f"**Employees:** {len(emp_df)}")
    st.caption(f"**Training records:** {len(training_df)}")
    st.caption(f"**Compliance records:** {total_certs}")
    st.caption(f"**Cert types tracked:** {n_cert_types}")

# ── Apply global filters ──────────────────────────────────────────────────────
def apply_compliance_filters(df):
    out = df.copy()
    if sel_emp != "All Employees":
        out = out[out["name"] == sel_emp]
    if sel_cert != "All Certifications":
        out = out[out["certification_name"] == sel_cert]
    if status_options:
        out = out[out["renewal_status"].isin(status_options)]
    return out

def apply_training_filters(df):
    out = df.copy()
    if sel_emp != "All Employees":
        out = out[out["name"] == sel_emp]
    if sel_station != "All Stations":
        out = out[out["station_code"] == sel_station]
    return out

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_comp, tab_train, tab_pipe, tab_profile = st.tabs([
    "🔴 Compliance Certs",
    "📊 Station Training",
    "🌱 Train-the-Trainer",
    "👤 Employee Profile",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMPLIANCE CERTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_comp:
    st.subheader("Compliance Certification Status")
    st.caption(
        "Alberta law requires every QSR employee to hold valid certifications before "
        "working a shift. 🟥 Expired = compliance gap · 🟨 Expiring Soon = renew within 60 days"
    )

    filtered_comp = apply_compliance_filters(compliance_df)

    # ── Action required — expired first ──────────────────────────────────────
    expired_recs = filtered_comp[filtered_comp["renewal_status"] == "Expired"].copy()
    soon_recs    = filtered_comp[filtered_comp["renewal_status"] == "Expiring Soon"].copy()

    if not expired_recs.empty:
        with st.expander(f"🚨 EXPIRED CERTIFICATIONS — {len(expired_recs)} records  (immediate action required)", expanded=True):
            for _, row in expired_recs.sort_values("days_until_expiry").iterrows():
                overdue = abs(int(row["days_until_expiry"]))
                st.error(
                    f"**{row['name']}** ({row['employee_id']})  ·  "
                    f"**{row['certification_name']}**  ·  "
                    f"Expired {overdue} days ago  ·  "
                    f"Issuer: {row['issuing_body']}  ·  "
                    f"Expiry: {row['expiry_date']}"
                )

    if not soon_recs.empty:
        with st.expander(f"⚠️ EXPIRING WITHIN 60 DAYS — {len(soon_recs)} records  (schedule renewal now)", expanded=True):
            for _, row in soon_recs.sort_values("days_until_expiry").iterrows():
                days_left = int(row["days_until_expiry"])
                st.warning(
                    f"**{row['name']}** ({row['employee_id']})  ·  "
                    f"**{row['certification_name']}**  ·  "
                    f"{days_left} days remaining  ·  "
                    f"Expires: {row['expiry_date']}  ·  "
                    f"Issuer: {row['issuing_body']}"
                )

    if expired_recs.empty and soon_recs.empty:
        st.success("✅ No expired or expiring-soon certifications match the current filter.")

    st.divider()

    # ── Full compliance table ─────────────────────────────────────────────────
    st.markdown("**Full Compliance Records**")

    display_comp = filtered_comp[[
        "employee_id", "name", "role", "certification_name",
        "issued_date", "expiry_date", "days_until_expiry", "renewal_status", "issuing_body"
    ]].copy()
    display_comp.columns = [
        "ID", "Name", "Role", "Certification",
        "Issued", "Expires", "Days Left", "Status", "Issuing Body"
    ]
    display_comp = display_comp.reset_index(drop=True)

    def style_compliance(row):
        style = STATUS_COLORS.get(row["Status"], "")
        return [style if col == "Status" else "" for col in row.index]

    st.dataframe(
        display_comp.style.apply(style_compliance, axis=1),
        width="stretch",
        hide_index=True,
        height=400,
    )
    st.caption(f"Showing {len(display_comp)} records · Filtered by: Employee={sel_emp}, Cert={sel_cert}, Status={status_options or 'none'}")

    st.divider()

    # ── Per-certification coverage breakdown ──────────────────────────────────
    st.markdown("**Coverage by Certification Type**  (all employees, unfiltered)")

    for cert_name in compliance_df["certification_name"].unique():
        cert_slice  = compliance_df[compliance_df["certification_name"] == cert_name]
        total_req   = len(cert_slice)
        n_curr      = (cert_slice["renewal_status"] == "Current").sum()
        n_exp_s     = (cert_slice["renewal_status"] == "Expiring Soon").sum()
        n_exp       = (cert_slice["renewal_status"] == "Expired").sum()
        coverage    = round((n_curr + n_exp_s) / total_req * 100) if total_req else 0
        gap_flag    = "  🚨 COMPLIANCE GAP" if n_exp > 0 else ""

        col_a, col_b = st.columns([3, 7])
        with col_a:
            st.markdown(f"**{cert_name}**{gap_flag}")
            st.caption(f"{n_curr} current · {n_exp_s} expiring · {n_exp} expired / {total_req} required")
        with col_b:
            st.progress(coverage / 100, text=f"{coverage}% compliant")

    st.divider()

    # ── Employee compliance scorecard ─────────────────────────────────────────
    st.markdown("**Employee Compliance Scorecard**")

    emp_scores = []
    for _, emp in emp_df.iterrows():
        emp_comp = compliance_df[compliance_df["employee_id"] == emp["emp_id"]]
        if emp_comp.empty:
            continue
        n_c = (emp_comp["renewal_status"] == "Current").sum()
        n_s = (emp_comp["renewal_status"] == "Expiring Soon").sum()
        n_e = (emp_comp["renewal_status"] == "Expired").sum()
        total = len(emp_comp)
        score = round((n_c + n_s * 0.5) / total * 100) if total else 0
        emp_scores.append({
            "ID":        emp["emp_id"],
            "Name":      emp["name"],
            "Role":      emp["role"],
            "Current":   int(n_c),
            "Soon":      int(n_s),
            "Expired":   int(n_e),
            "Total":     int(total),
            "Score %":   score,
            "Status":    "🚨 Has Gaps" if n_e > 0 else ("⚠️ Renewal Due" if n_s > 0 else "✅ Compliant"),
        })

    score_df = pd.DataFrame(emp_scores).sort_values("Expired", ascending=False)

    def style_score(row):
        if row["Expired"] > 0:
            return ["background-color:#f8d7da"] * len(row)
        if row["Soon"] > 0:
            return ["background-color:#fff3cd"] * len(row)
        return ["background-color:#d4edda"] * len(row)

    st.dataframe(
        score_df.style.apply(style_score, axis=1),
        width="stretch",
        hide_index=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATION TRAINING
# ═════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Station Proficiency Matrix")
    st.caption(
        "Proficiency scale: **0** Not Trained · **1** Basic · **2** Competent · "
        "**3** Proficient · **4** Trainer  ·  Source: employee records + training assessments"
    )

    # ── Proficiency heatmap ───────────────────────────────────────────────────
    station_cols = [f"station_{s}" for s in STATIONS]
    matrix_df    = emp_df[["emp_id", "name", "role"] + station_cols].copy()
    matrix_df.columns = ["ID", "Name", "Role"] + STATIONS

    # Color each cell by proficiency level
    def style_proficiency(val):
        if pd.isna(val):
            return ""
        v = int(val)
        bg = PROFICIENCY_COLORS.get(v, "#dee2e6")
        fg = "#ffffff" if v == 4 else "#000000"
        return f"background-color:{bg};color:{fg};text-align:center;font-weight:{'bold' if v==4 else 'normal'}"

    display_matrix = matrix_df.copy()
    for s in STATIONS:
        display_matrix[s] = display_matrix[s].apply(
            lambda v: f"{int(v)} — {PROFICIENCY_LABELS[int(v)]}" if not pd.isna(v) else ""
        )

    # Use raw numbers for styling, labels for display
    numeric_matrix = matrix_df[STATIONS].copy()

    st.dataframe(
        matrix_df.rename(columns={s: STATION_LABELS[s] for s in STATIONS})
        .drop(columns="ID")
        .style.map(style_proficiency, subset=[STATION_LABELS[s] for s in STATIONS]),
        width="stretch",
        hide_index=True,
    )

    # Legend
    leg_cols = st.columns(5)
    for i, (level, label) in enumerate(PROFICIENCY_LABELS.items()):
        bg = PROFICIENCY_COLORS[level]
        fg = "#fff" if level == 4 else "#000"
        leg_cols[i].markdown(
            f"<div style='background:{bg};color:{fg};padding:6px 10px;border-radius:6px;"
            f"text-align:center;font-size:0.82rem'><b>{level}</b> — {label}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Station coverage summary ──────────────────────────────────────────────
    st.markdown("**Station Coverage Summary**  (employees at each proficiency tier)")

    cov_cols = st.columns(len(STATIONS))
    for i, s in enumerate(STATIONS):
        col_data = emp_df[f"station_{s}"]
        trained  = (col_data >= 1).sum()
        profic   = (col_data >= 3).sum()
        trainers = (col_data == 4).sum()
        cov_cols[i].markdown(
            f"**{s}**\n\n"
            f"Trained: **{trained}**\n\n"
            f"Proficient+: **{profic}**\n\n"
            f"Trainers: **{trainers}**"
        )

    st.divider()

    # ── Training records table ────────────────────────────────────────────────
    st.markdown("**Training Records**")

    filtered_train = apply_training_filters(training_df)

    display_train = filtered_train[[
        "record_id", "employee_id", "name", "role", "station_code",
        "training_type", "training_date", "score", "proficiency_level",
        "status", "certification_date", "expiry_date",
    ]].copy()
    display_train.columns = [
        "Record", "ID", "Name", "Role", "Station",
        "Training Type", "Date", "Score", "Level",
        "Status", "Cert Date", "Expiry",
    ]
    display_train = display_train.reset_index(drop=True)

    def style_training_status(row):
        v = row["Status"]
        if v == "Pending Renewal": return ["background-color:#fff3cd"] * len(row)
        if v == "Completed":       return ["background-color:#d4edda"] * len(row)
        if v == "In Progress":     return ["background-color:#cfe2ff"] * len(row)
        return [""] * len(row)

    def style_level(val):
        bg = PROFICIENCY_COLORS.get(int(val) if not pd.isna(val) else 0, "")
        fg = "#fff" if int(val) == 4 else "#000"
        return f"background-color:{bg};color:{fg};text-align:center"

    st.dataframe(
        display_train.style
            .apply(style_training_status, axis=1)
            .map(style_level, subset=["Level"]),
        width="stretch",
        hide_index=True,
        height=420,
    )
    st.caption(f"Showing {len(display_train)} records · Station={sel_station} · Employee={sel_emp}")

    # ── Score distribution by station ────────────────────────────────────────
    st.divider()
    st.markdown("**Average Training Score by Station**")
    avg_scores = (
        training_df[training_df["score"].notna()]
        .groupby("station_code")["score"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"station_code": "Station", "score": "Avg Score"})
        .sort_values("Avg Score", ascending=False)
    )
    st.bar_chart(avg_scores.set_index("Station"), height=220)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAIN-THE-TRAINER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
with tab_pipe:
    st.subheader("Train-the-Trainer Pipeline")
    st.caption(
        "Five-stage pipeline from newly trained crew to certified station trainer. "
        "Stage is determined by highest completed training type per employee × station."
    )

    # ── Pipeline legend ───────────────────────────────────────────────────────
    stages = ["Identified", "Shadow", "Co-Train", "Lead", "Mentoring"]
    stage_desc = {
        "Identified": "Promotion-ready crew flagged for trainer track",
        "Shadow":     "Observing certified trainer during live shifts",
        "Co-Train":   "Training alongside a trainer, taking partial lead",
        "Lead":       "Running station independently, trainer observing",
        "Mentoring":  "Certified trainer — developing the next generation",
    }
    stage_cols = st.columns(5)
    for i, stage in enumerate(stages):
        bg, fg = STAGE_COLORS[stage]
        stage_cols[i].markdown(
            f"<div style='background:{bg};color:{fg};padding:8px 6px;border-radius:8px;"
            f"text-align:center;font-size:0.82rem'>"
            f"<b>Stage {i+1}</b><br>{stage}<br>"
            f"<span style='font-size:0.74rem;opacity:0.85'>{stage_desc[stage]}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Derive pipeline stage per employee × station ──────────────────────────
    # Stage 5 Mentoring  = Trainer Certification, level=4, Completed
    # Stage 4 Lead       = Proficiency Assessment, level=3, Completed
    # Stage 3 Co-Train   = Competency Check, level=2, Completed
    # Stage 2 Shadow     = Initial Training, level=1, Completed
    # Stage 1 Identified = promotion_ready=1 or bench_priority>0, not yet in pipeline

    TYPE_TO_STAGE = {
        "Initial Training":       "Shadow",
        "Competency Check":       "Co-Train",
        "Proficiency Assessment":  "Lead",
        "Trainer Certification":  "Mentoring",
    }
    STAGE_ORDER = {"Shadow": 2, "Co-Train": 3, "Lead": 4, "Mentoring": 5}

    completed_train = training_df[training_df["status"] == "Completed"].copy()
    completed_train["stage"] = completed_train["training_type"].map(TYPE_TO_STAGE)
    completed_train["stage_order"] = completed_train["stage"].map(STAGE_ORDER)

    # Highest completed stage per emp × station
    best_stage = (
        completed_train
        .sort_values("stage_order", ascending=False)
        .drop_duplicates(subset=["employee_id", "station_code"], keep="first")
        [["employee_id", "name", "role", "station_code", "stage", "stage_order"]]
        .reset_index(drop=True)
    )

    # Add "Identified" for promotion-ready employees who appear in no station pipeline yet
    promo_ready = emp_df[(emp_df["promotion_ready"] == 1) | (emp_df["bench_priority"] > 0)][["emp_id","name","role"]].copy()
    promo_ready.columns = ["employee_id","name","role"]
    identified_rows = []
    for _, pr in promo_ready.iterrows():
        # Check if they're already in the pipeline for any station
        in_pipeline = best_stage[best_stage["employee_id"] == pr["employee_id"]]
        if in_pipeline.empty:
            # Add as Identified across all stations they have any training on
            emp_stations = training_df[training_df["employee_id"] == pr["employee_id"]]["station_code"].unique()
            for s in emp_stations:
                identified_rows.append({
                    "employee_id": pr["employee_id"], "name": pr["name"], "role": pr["role"],
                    "station_code": s, "stage": "Identified", "stage_order": 1,
                })
    if identified_rows:
        best_stage = pd.concat([best_stage, pd.DataFrame(identified_rows)], ignore_index=True)

    # ── Pipeline by station ───────────────────────────────────────────────────
    st.markdown("**Pipeline View by Station**")

    for station in STATIONS:
        station_pipe = best_stage[best_stage["station_code"] == station].sort_values("stage_order", ascending=False)
        if station_pipe.empty:
            continue

        mentors    = station_pipe[station_pipe["stage"] == "Mentoring"]
        non_mentor = station_pipe[station_pipe["stage"] != "Mentoring"]

        with st.expander(
            f"**{station}** — {STATION_LABELS[station]}  ·  "
            f"{len(mentors)} certified trainer(s)  ·  "
            f"{len(non_mentor)} in pipeline",
            expanded=(station in ["BEV", "PREP", "SUP"]),
        ):
            badge_html = ""
            for _, row in station_pipe.iterrows():
                bg, fg = STAGE_COLORS.get(row["stage"], ("#ccc", "#000"))
                badge_html += (
                    f"<span class='stage-badge' style='background:{bg};color:{fg}'>"
                    f"{row['stage']} — {row['name']} ({row['role']})</span> "
                )
            st.markdown(badge_html, unsafe_allow_html=True)

            if len(mentors) == 0:
                st.error(f"⚠️ **No certified trainer for {station}** — training continuity at risk.")
            elif len(mentors) == 1:
                st.warning(f"Only **1 certified trainer** for {station}. Succession risk if this person leaves.")

    st.divider()

    # ── Overall pipeline funnel ───────────────────────────────────────────────
    st.markdown("**Overall Pipeline Funnel (all stations combined)**")

    funnel_counts = best_stage.groupby("stage").size().reindex(
        ["Identified", "Shadow", "Co-Train", "Lead", "Mentoring"], fill_value=0
    ).reset_index()
    funnel_counts.columns = ["Stage", "Records"]

    col_f, col_t = st.columns([2, 3])
    with col_f:
        for _, row in funnel_counts.iterrows():
            bg, fg = STAGE_COLORS.get(row["Stage"], ("#ccc", "#000"))
            bar_w  = max(int(row["Records"] / funnel_counts["Records"].max() * 100), 8)
            st.markdown(
                f"<div style='margin:4px 0'>"
                f"<span style='display:inline-block;width:110px;font-size:0.85rem'>{row['Stage']}</span>"
                f"<span style='display:inline-block;width:{bar_w}%;background:{bg};color:{fg};"
                f"padding:4px 8px;border-radius:4px;font-weight:600'>{row['Records']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_t:
        st.caption("**What each count means:**")
        for stage, desc in stage_desc.items():
            st.caption(f"• **{stage}:** {desc}")

    st.divider()

    # ── Trainer roster ────────────────────────────────────────────────────────
    st.markdown("**Certified Trainer Roster**")
    st.caption("Employees with Trainer Certification (level 4) by station")

    trainer_df = (
        best_stage[best_stage["stage"] == "Mentoring"]
        .groupby(["employee_id", "name", "role"])["station_code"]
        .apply(lambda x: " · ".join(sorted(x)))
        .reset_index()
        .rename(columns={"station_code": "Certified Stations"})
    )
    trainer_df.columns = ["ID", "Name", "Role", "Certified Stations"]

    if trainer_df.empty:
        st.info("No trainer certifications found in training records.")
    else:
        st.dataframe(trainer_df, width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EMPLOYEE PROFILE
# ═════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.subheader("Employee Training & Compliance Profile")

    emp_list = sorted(emp_df["name"].tolist())
    sel_profile_name = st.selectbox("Select employee", emp_list, key="profile_selector")

    sel_profile = emp_df[emp_df["name"] == sel_profile_name].iloc[0]
    emp_id = sel_profile["emp_id"]

    # ── Employee info header ──────────────────────────────────────────────────
    col_info, col_career = st.columns([2, 2])

    with col_info:
        st.markdown(f"### {sel_profile['name']}")
        st.markdown(f"**Role:** {sel_profile['role']}")
        st.markdown(f"**ID:** {emp_id}")
        st.markdown(f"**Hours:** {sel_profile['min_hours']}–{sel_profile['max_hours']} hrs/wk  ·  Max {sel_profile['max_days_per_week']} days")
        st.markdown(f"**Availability:** {int(sel_profile['availability_start']):02d}:00 – {int(sel_profile['availability_end']):02d}:00")
        st.markdown(f"**Day off:** {sel_profile['days_off']}")
        is_trainer_label = "✅ Certified Trainer" if sel_profile["is_trainer"] == 1 else "Not on trainer roster"
        st.markdown(f"**Trainer status:** {is_trainer_label}")

    with col_career:
        st.markdown("**Career Path**")
        st.markdown(f"**Current role:** {sel_profile['role']}")
        st.markdown(f"**Next role:** {sel_profile['next_role'] or '—'}")
        promo = "✅ Yes" if sel_profile["promotion_ready"] == 1 else "Not yet"
        bench = int(sel_profile["bench_priority"])
        bench_label = {0: "Not on bench", 1: "Priority 1 (next up)", 2: "Priority 2"}.get(bench, str(bench))
        st.markdown(f"**Promotion ready:** {promo}")
        st.markdown(f"**Bench priority:** {bench_label}")

    st.divider()

    # ── Station proficiency for this employee ─────────────────────────────────
    st.markdown("**Station Proficiency**")

    prof_cols = st.columns(len(STATIONS))
    for i, s in enumerate(STATIONS):
        level = int(sel_profile[f"station_{s}"])
        bg    = PROFICIENCY_COLORS[level]
        fg    = "#fff" if level == 4 else "#000"
        prof_cols[i].markdown(
            f"<div style='background:{bg};color:{fg};padding:8px 4px;border-radius:8px;"
            f"text-align:center;font-size:0.82rem'>"
            f"<b>{s}</b><br>Level {level}<br>"
            f"<span style='font-size:0.74rem'>{PROFICIENCY_LABELS[level]}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Compliance certs for this employee ────────────────────────────────────
    st.markdown("**Compliance Certifications**")

    emp_comp = compliance_df[compliance_df["employee_id"] == emp_id].copy()

    if emp_comp.empty:
        st.info("No compliance records found for this employee.")
    else:
        n_curr_e = (emp_comp["renewal_status"] == "Current").sum()
        n_soon_e = (emp_comp["renewal_status"] == "Expiring Soon").sum()
        n_exp_e  = (emp_comp["renewal_status"] == "Expired").sum()

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Current",      int(n_curr_e))
        cc2.metric("Expiring Soon", int(n_soon_e))
        cc3.metric("Expired",      int(n_exp_e))

        display_emp_comp = emp_comp[[
            "certification_name", "issued_date", "expiry_date",
            "days_until_expiry", "renewal_status", "issuing_body"
        ]].copy()
        display_emp_comp.columns = ["Certification", "Issued", "Expires", "Days Left", "Status", "Issuing Body"]

        def style_emp_comp(row):
            style = STATUS_COLORS.get(row["Status"], "")
            return [style if col == "Status" else "" for col in row.index]

        st.dataframe(
            display_emp_comp.reset_index(drop=True).style.apply(style_emp_comp, axis=1),
            width="stretch",
            hide_index=True,
        )

    st.divider()

    # ── Training records for this employee ────────────────────────────────────
    st.markdown("**Training Records**")

    emp_train = training_df[training_df["employee_id"] == emp_id].copy()

    if emp_train.empty:
        st.info("No training records found for this employee.")
    else:
        display_emp_train = emp_train[[
            "record_id", "station_code", "training_type", "training_date",
            "score", "proficiency_level", "status", "certification_date", "expiry_date"
        ]].copy()
        display_emp_train.columns = [
            "Record", "Station", "Type", "Date",
            "Score", "Level", "Status", "Cert Date", "Expiry"
        ]

        def style_emp_train(row):
            v = row["Status"]
            if v == "Pending Renewal": return ["background-color:#fff3cd"] * len(row)
            if v == "Completed":       return ["background-color:#d4edda"] * len(row)
            if v == "In Progress":     return ["background-color:#cfe2ff"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_emp_train.reset_index(drop=True).style.apply(style_emp_train, axis=1),
            width="stretch",
            hide_index=True,
        )

        # Score trend for this employee if multiple records
        scored = emp_train[emp_train["score"].notna()].sort_values("training_date")
        if len(scored) >= 3:
            st.divider()
            st.markdown("**Assessment Score History**")
            score_chart = scored[["training_date", "score", "station_code"]].copy()
            score_chart["label"] = score_chart["station_code"] + " · " + score_chart["training_date"].str[:10]
            st.line_chart(score_chart.set_index("label")[["score"]], height=200)
