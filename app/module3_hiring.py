"""
Module 3 — Hiring Needs & Team Morale Dashboard
Run with: streamlit run app/module3_hiring.py
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
    page_title="Hiring & Morale — QSR Ops",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.trigger-high   { background:#f8d7da; border-left:5px solid #dc3545;
                  border-radius:6px; padding:10px 14px; margin:6px 0; }
.trigger-medium { background:#fff3cd; border-left:5px solid #ffc107;
                  border-radius:6px; padding:10px 14px; margin:6px 0; }
.trigger-low    { background:#d4edda; border-left:5px solid #28a745;
                  border-radius:6px; padding:10px 14px; margin:6px 0; }
.morale-risk    { background:#f8d7da; border-radius:8px; padding:12px; margin:4px 0; }
.morale-warn    { background:#fff3cd; border-radius:8px; padding:12px; margin:4px 0; }
.morale-ok      { background:#d4edda; border-radius:8px; padding:12px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    emp_df     = pd.read_csv(os.path.join(ROOT, "data", "qsr_employees.csv"))
    hiring_df  = pd.read_csv(os.path.join(ROOT, "data", "qsr_hiring_analysis.csv"))
    survey_df  = pd.read_csv(os.path.join(ROOT, "data", "qsr_employee_survey.csv"))
    sales_df   = pd.read_csv(os.path.join(ROOT, "data", "qsr_sales_data.csv"))
    return emp_df, hiring_df, survey_df, sales_df

emp_df, hiring_df, survey_df, sales_df = load_data()

# Merge employee names into survey
emp_names = emp_df[["emp_id", "name"]].rename(columns={"emp_id": "employee_id"})
survey_df = survey_df.merge(emp_names, on="employee_id", how="left")

# ── Constants ─────────────────────────────────────────────────────────────────
TRIGGER_INFO = {
    "T1": ("SUP Coverage Gap",        "Only 1 employee certified at SUP Level 2+. Single point of failure for supervisor shifts."),
    "T2": ("Supervisor Day Coverage", "A weekday has fewer than 2 supervisors available — scheduling risk."),
    "T3": ("Demand Surge",            "2+ consecutive weeks of demand >20% above annual average."),
    "T4": ("At-Risk Employee",        "Employee morale index below 3.0 for 3+ consecutive weeks — turnover signal."),
    "T5": ("Role Headcount Gap",      "A role (e.g., Manager) falls below minimum required headcount."),
    "T6": ("Capacity Overload",       "Weekly staffing hours demanded exceeds 90% of total team capacity."),
    "T7": ("Succession Vacancy",      "A promotion-ready employee advances, leaving a downstream vacancy."),
}
MORALE_FACTORS = [
    "hours_consistency", "comms_training", "manager_floor_presence",
    "product_availability", "personal_time_respect", "daily_plan_clarity",
]
MORALE_LABELS = {
    "hours_consistency":      "Hours Consistency",
    "comms_training":         "Comms & Training",
    "manager_floor_presence": "Manager Floor Presence",
    "product_availability":   "Product Availability",
    "personal_time_respect":  "Personal Time Respect",
    "daily_plan_clarity":     "Daily Plan Clarity",
}
MORALE_WEIGHTS = {
    "hours_consistency":      0.20,
    "comms_training":         0.15,
    "manager_floor_presence": 0.15,
    "product_availability":   0.15,
    "personal_time_respect":  0.20,
    "daily_plan_clarity":     0.15,
}
SEV_ORDER  = {"High": 0, "Medium": 1, "Low": 2}
SEV_COLORS = {"High": "#f8d7da", "Medium": "#fff3cd", "Low": "#d4edda"}
SEV_ICONS  = {"High": "🚨", "Medium": "⚠️", "Low": "💡"}

# Weekly demand from sales data
weekly_demand = (
    sales_df.groupby("iso_week")["staff_needed"].sum().reset_index()
    .rename(columns={"iso_week": "week_number", "staff_needed": "total_hrs_needed"})
)
ANNUAL_AVG_DEMAND = weekly_demand["total_hrs_needed"].mean()
TEAM_CAPACITY_90  = 311  # 90% of effective weekly capacity (from stage3 analysis)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Hiring Needs & Team Morale Dashboard")
st.caption(
    "7 programmatic hiring triggers · team morale tracking · at-risk alerts · "
    "capacity gap analysis  ·  Data snapshot: 2024 full year"
)
st.divider()

# ── Top KPI row ───────────────────────────────────────────────────────────────
total_triggers  = len(hiring_df)
n_high          = (hiring_df["severity"] == "High").sum()
n_medium        = (hiring_df["severity"] == "Medium").sum()
n_low           = (hiring_df["severity"] == "Low").sum()
total_headcount = hiring_df["estimated_headcount"].sum()
avg_morale      = survey_df["morale_index"].mean()
at_risk_emps    = survey_df[survey_df["at_risk_flag"] == 1]["employee_id"].nunique()

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Hiring Triggers",    total_triggers)
c2.metric("🚨 High Severity",   n_high)
c3.metric("⚠️ Medium Severity",  n_medium)
c4.metric("💡 Low Severity",     n_low)
c5.metric("Headcount Needed",   int(total_headcount))
c6.metric("Team Avg Morale",    f"{avg_morale:.2f} / 5.0")
c7.metric("At-Risk Employees",  at_risk_emps, delta=f"{at_risk_emps} flagged", delta_color="inverse")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")

    sev_filter = st.multiselect(
        "Severity", ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    trigger_filter = st.multiselect(
        "Trigger Code",
        sorted(hiring_df["trigger_code"].unique().tolist()),
        default=sorted(hiring_df["trigger_code"].unique().tolist()),
    )

    emp_options = ["All Employees"] + sorted(emp_df["name"].tolist())
    sel_emp     = st.selectbox("Employee (Morale)", emp_options)

    st.divider()
    st.caption("**Trigger legend:**")
    for code, (label, _) in TRIGGER_INFO.items():
        st.caption(f"**{code}** — {label}")

    st.divider()
    st.caption(f"**Annual avg demand:** {ANNUAL_AVG_DEMAND:.0f} hrs/wk")
    st.caption(f"**Team capacity (90%):** {TEAM_CAPACITY_90} hrs/wk")
    st.caption(f"**Structural gap:** {ANNUAL_AVG_DEMAND - TEAM_CAPACITY_90:.0f} hrs/wk")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_hire, tab_cap, tab_morale, tab_profile = st.tabs([
    "🚨 Hiring Triggers",
    "📈 Demand & Capacity",
    "💚 Team Morale",
    "👤 Employee Morale Profile",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — HIRING TRIGGERS
# ═════════════════════════════════════════════════════════════════════════════
with tab_hire:
    st.subheader("Active Hiring Triggers — 2024 Analysis")
    st.caption(
        "Each trigger represents a data-driven signal that a hire or internal action is needed. "
        "Sorted by severity then week number."
    )

    filtered_hiring = hiring_df[
        hiring_df["severity"].isin(sev_filter) &
        hiring_df["trigger_code"].isin(trigger_filter)
    ].sort_values(["severity", "week_number"], key=lambda s: s.map(SEV_ORDER) if s.name == "severity" else s)

    # ── Priority action cards ─────────────────────────────────────────────────
    high_recs = filtered_hiring[filtered_hiring["severity"] == "High"]
    if not high_recs.empty:
        st.markdown("### 🚨 High Priority — Immediate Action")
        for _, row in high_recs.iterrows():
            tcode = row["trigger_code"]
            tlabel, tdesc = TRIGGER_INFO.get(tcode, (tcode, ""))
            hc_text = f"{int(row['estimated_headcount'])} hire(s)" if pd.notna(row["estimated_headcount"]) else ""
            gap_text = f"{int(row['hrs_gap_per_week'])} hrs/wk gap" if pd.notna(row["hrs_gap_per_week"]) and row["hrs_gap_per_week"] > 0 else ""
            st.error(
                f"**{tcode} — {tlabel}**  ·  {row['affected_role']}  ·  {hc_text}  ·  {gap_text}\n\n"
                f"*{row['trigger_label']}*\n\n"
                f"**Action:** {row['recommended_action']}"
            )

    medium_recs = filtered_hiring[filtered_hiring["severity"] == "Medium"]
    if not medium_recs.empty:
        st.markdown("### ⚠️ Medium Priority — Schedule Action")
        for _, row in medium_recs.iterrows():
            tcode = row["trigger_code"]
            tlabel, _ = TRIGGER_INFO.get(tcode, (tcode, ""))
            hc_text  = f"{int(row['estimated_headcount'])} hire(s)" if pd.notna(row["estimated_headcount"]) else ""
            gap_text = f"Wk {int(row['week_number'])}  ·  {int(row['hrs_gap_per_week'])} hrs gap" if pd.notna(row["hrs_gap_per_week"]) and row["hrs_gap_per_week"] > 0 else f"Wk {int(row['week_number'])}"
            st.warning(
                f"**{tcode} — {tlabel}**  ·  {row['affected_role']}  ·  {hc_text}  ·  {gap_text}\n\n"
                f"**Action:** {row['recommended_action']}"
            )

    low_recs = filtered_hiring[filtered_hiring["severity"] == "Low"]
    if not low_recs.empty:
        with st.expander(f"💡 Low Priority — {len(low_recs)} succession / pipeline items"):
            for _, row in low_recs.iterrows():
                tcode = row["trigger_code"]
                tlabel, _ = TRIGGER_INFO.get(tcode, (tcode, ""))
                st.info(
                    f"**{tcode} — {tlabel}**  ·  {row['affected_role']}  ·  "
                    f"{int(row['estimated_headcount'])} position(s)\n\n"
                    f"**Action:** {row['recommended_action']}"
                )

    st.divider()

    # ── Full trigger table ────────────────────────────────────────────────────
    st.markdown("**All Hiring Trigger Records**")

    display_hire = filtered_hiring[[
        "analysis_id", "trigger_code", "severity", "affected_role",
        "affected_station", "availability_needed", "hrs_gap_per_week",
        "estimated_headcount", "status", "recommended_action"
    ]].copy()
    display_hire.columns = [
        "ID", "Trigger", "Severity", "Role",
        "Station", "Availability", "Hrs Gap/Wk",
        "Headcount", "Status", "Recommended Action"
    ]
    display_hire = display_hire.reset_index(drop=True)

    def style_severity(row):
        bg = SEV_COLORS.get(row["Severity"], "")
        return [f"background-color:{bg}" if bg else ""] * len(row)

    st.dataframe(
        display_hire.style.apply(style_severity, axis=1),
        width="stretch",
        hide_index=True,
        height=380,
    )

    st.divider()

    # ── Trigger breakdown ─────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Triggers by Code**")
        tcode_counts = hiring_df.groupby("trigger_code").agg(
            Count=("analysis_id", "count"),
            Headcount=("estimated_headcount", "sum"),
        ).reset_index()
        tcode_counts["Label"] = tcode_counts["trigger_code"].map(
            lambda c: TRIGGER_INFO.get(c, (c, ""))[0]
        )
        tcode_counts["Display"] = tcode_counts["trigger_code"] + " — " + tcode_counts["Label"]
        st.dataframe(
            tcode_counts[["Display", "Count", "Headcount"]].rename(
                columns={"Display": "Trigger", "Count": "Records", "Headcount": "Hires Needed"}
            ),
            width="stretch", hide_index=True,
        )

    with col_b:
        st.markdown("**Headcount Needed by Role**")
        role_counts = (
            hiring_df.groupby("affected_role")["estimated_headcount"]
            .sum()
            .reset_index()
            .rename(columns={"affected_role": "Role", "estimated_headcount": "Headcount"})
            .sort_values("Headcount", ascending=False)
        )
        st.bar_chart(role_counts.set_index("Role"), height=240)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMAND & CAPACITY
# ═════════════════════════════════════════════════════════════════════════════
with tab_cap:
    st.subheader("Weekly Demand vs Team Capacity — 2024")
    st.caption(
        "Annual average demand (427 hrs/wk) exceeds 90% team capacity (311 hrs/wk) every single week. "
        "The store is structurally under-resourced — this is the root cause of the T6 systemic trigger."
    )

    # ── KPI strip ────────────────────────────────────────────────────────────
    peak_week    = weekly_demand.loc[weekly_demand["total_hrs_needed"].idxmax()]
    low_week     = weekly_demand.loc[weekly_demand["total_hrs_needed"].idxmin()]
    demand_75th  = weekly_demand["total_hrs_needed"].quantile(0.75)

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("Annual Avg Demand",   f"{ANNUAL_AVG_DEMAND:.0f} hrs/wk")
    cc2.metric("Team Capacity (90%)", f"{TEAM_CAPACITY_90} hrs/wk")
    cc3.metric("Structural Gap",      f"{ANNUAL_AVG_DEMAND - TEAM_CAPACITY_90:.0f} hrs/wk",
               delta="every week", delta_color="inverse")
    cc4.metric("Peak Week Demand",    f"Wk {int(peak_week['week_number'])} — {int(peak_week['total_hrs_needed'])} hrs")
    cc5.metric("T6 Threshold (75th)", f"{demand_75th:.0f} hrs/wk")

    st.divider()

    # ── 52-week demand chart ──────────────────────────────────────────────────
    st.markdown("**Weekly Staffing Hours Demanded vs Capacity Ceiling (all 52 weeks)**")
    st.caption("🔴 T6 Peak weeks = top 25% demand. 🟦 Capacity line = 311 hrs (90% of team max).")

    chart_df = weekly_demand.copy().set_index("week_number")
    chart_df["Capacity (90%)"]     = TEAM_CAPACITY_90
    chart_df["Annual Avg Demand"]  = ANNUAL_AVG_DEMAND
    chart_df = chart_df.rename(columns={"total_hrs_needed": "Weekly Hrs Demanded"})

    st.line_chart(chart_df[["Weekly Hrs Demanded", "Capacity (90%)", "Annual Avg Demand"]], height=300)

    # ── T6 peak weeks table ───────────────────────────────────────────────────
    t6_weeks = hiring_df[hiring_df["trigger_code"] == "T6"].copy()
    t6_peak  = t6_weeks[t6_weeks["week_number"] > 1]  # exclude systemic row (week 1)

    if not t6_peak.empty:
        st.divider()
        st.markdown(f"**T6 Peak-Demand Weeks — {len(t6_peak)} weeks in top 25% demand**")
        t6_with_demand = t6_peak.merge(weekly_demand, on="week_number", how="left")
        t6_display = t6_with_demand[[
            "week_number", "total_hrs_needed", "hrs_gap_per_week",
            "estimated_headcount", "recommended_action"
        ]].copy()
        t6_display.columns = ["Week", "Hrs Demanded", "Gap vs Capacity", "Add'l Hires", "Action"]
        t6_display = t6_display.sort_values("Hrs Demanded", ascending=False).reset_index(drop=True)
        st.dataframe(t6_display, width="stretch", hide_index=True, height=300)

    st.divider()

    # ── Capacity breakdown by employee ────────────────────────────────────────
    st.markdown("**Current Team Capacity Breakdown**")
    st.caption("Effective weekly hours = max_hours × (max_days_per_week / 7)")

    cap_df = emp_df[["emp_id", "name", "role", "max_hours", "max_days_per_week"]].copy()
    cap_df["Effective Hrs/Wk"] = (
        cap_df["max_hours"] * (cap_df["max_days_per_week"] / 7)
    ).round(1)
    cap_df = cap_df.rename(columns={
        "emp_id": "ID", "name": "Name", "role": "Role",
        "max_hours": "Max Hrs", "max_days_per_week": "Max Days",
    })
    cap_df = cap_df.sort_values("Effective Hrs/Wk", ascending=False).reset_index(drop=True)

    col_cap, col_bar = st.columns([3, 2])
    with col_cap:
        st.dataframe(cap_df, width="stretch", hide_index=True)
        total_eff = cap_df["Effective Hrs/Wk"].sum()
        st.caption(
            f"Total effective capacity: **{total_eff:.0f} hrs/wk**  ·  "
            f"90% ceiling: **{total_eff * 0.90:.0f} hrs/wk**"
        )
    with col_bar:
        st.markdown("**Effective Hrs/Wk by Employee**")
        st.bar_chart(cap_df.set_index("Name")[["Effective Hrs/Wk"]], height=360)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TEAM MORALE
# ═════════════════════════════════════════════════════════════════════════════
with tab_morale:
    st.subheader("Team Morale — 2024 Full Year")
    st.caption(
        "Weekly morale index (1–5) per employee, tracked across all 52 weeks. "
        "At-risk flag fires when morale_index < 3.0 for 3+ consecutive weeks — a turnover signal."
    )

    # ── KPI strip ────────────────────────────────────────────────────────────
    team_avg   = survey_df["morale_index"].mean()
    worst_week = survey_df.groupby("week_number")["morale_index"].mean().idxmin()
    best_week  = survey_df.groupby("week_number")["morale_index"].mean().idxmax()
    lowest_factor = survey_df[MORALE_FACTORS].mean().idxmin()

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Team Avg Morale",      f"{team_avg:.2f} / 5.0")
    mc2.metric("At-Risk Employees",    at_risk_emps, delta=f"{at_risk_emps} need attention", delta_color="inverse")
    mc3.metric("Lowest Morale Factor", MORALE_LABELS[lowest_factor],
               delta="most at-risk dimension", delta_color="inverse")
    mc4.metric("Best / Worst Week",    f"Wk {best_week} / Wk {worst_week}")

    st.divider()

    # ── At-risk alert cards ───────────────────────────────────────────────────
    at_risk_df = survey_df[survey_df["at_risk_flag"] == 1]
    if not at_risk_df.empty:
        st.markdown("### 🚨 At-Risk Employee Alerts")
        for emp_id in at_risk_df["employee_id"].unique():
            emp_risk  = at_risk_df[at_risk_df["employee_id"] == emp_id]
            emp_name  = emp_risk["name"].iloc[0]
            emp_role  = emp_risk["role"].iloc[0]
            n_weeks   = len(emp_risk)
            worst_mi  = emp_risk["morale_index"].min()
            worst_wk  = emp_risk.loc[emp_risk["morale_index"].idxmin(), "week_number"]
            avg_ret   = emp_risk["retention_intent"].mean()
            st.error(
                f"**{emp_name}** ({emp_id}) — {emp_role}\n\n"
                f"**{n_weeks} consecutive at-risk weeks**  ·  "
                f"Worst morale: **{worst_mi:.2f}** (Week {int(worst_wk)})  ·  "
                f"Avg retention intent: **{avg_ret:.1f} / 5.0**\n\n"
                f"This employee triggered a **T4 hiring flag**. "
                f"Recommended: manager 1-on-1, schedule review, workload assessment."
            )

    st.divider()

    # ── Weekly morale trend ───────────────────────────────────────────────────
    st.markdown("**Weekly Team Morale Index — All 52 Weeks**")
    st.caption("Average morale across all 15 employees. Dip below 3.0 = at-risk territory.")

    weekly_morale = survey_df.groupby("week_number")["morale_index"].mean().reset_index()
    weekly_morale.columns = ["Week", "Avg Morale Index"]
    weekly_morale["At-Risk Line (3.0)"] = 3.0
    st.line_chart(weekly_morale.set_index("Week"), height=250)

    st.divider()

    # ── 6 morale factors breakdown ────────────────────────────────────────────
    st.markdown("**Morale Factor Breakdown — Annual Averages**")
    st.caption(
        f"Weights: hours_consistency & personal_time_respect = 0.20 each (highest).  "
        f"All others = 0.15.  Lowest factor: **{MORALE_LABELS[lowest_factor]}**"
    )

    factor_avgs = survey_df[MORALE_FACTORS].mean().reset_index()
    factor_avgs.columns = ["Factor", "Avg Score"]
    factor_avgs["Label"]  = factor_avgs["Factor"].map(MORALE_LABELS)
    factor_avgs["Weight"] = factor_avgs["Factor"].map(MORALE_WEIGHTS)
    factor_avgs["Weighted Impact"] = (factor_avgs["Avg Score"] * factor_avgs["Weight"]).round(3)
    factor_avgs = factor_avgs.sort_values("Avg Score")

    col_fbar, col_ftable = st.columns([3, 2])
    with col_fbar:
        st.bar_chart(factor_avgs.set_index("Label")[["Avg Score"]], height=260)
    with col_ftable:
        display_factors = factor_avgs[["Label", "Avg Score", "Weight", "Weighted Impact"]].copy()
        display_factors["Avg Score"] = display_factors["Avg Score"].round(2)
        display_factors = display_factors.sort_values("Avg Score").reset_index(drop=True)

        def style_factor(row):
            if row["Avg Score"] == display_factors["Avg Score"].min():
                return ["background-color:#f8d7da"] * len(row)
            if row["Avg Score"] == display_factors["Avg Score"].max():
                return ["background-color:#d4edda"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_factors.rename(columns={"Label": "Factor"})
            .style.apply(style_factor, axis=1),
            width="stretch", hide_index=True,
        )

    st.divider()

    # ── Per-role morale comparison ────────────────────────────────────────────
    st.markdown("**Morale by Role — Annual Average**")

    role_morale = (
        survey_df.groupby("role")["morale_index"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"role": "Role", "morale_index": "Avg Morale"})
        .sort_values("Avg Morale", ascending=False)
    )

    rcol1, rcol2 = st.columns([2, 3])
    with rcol1:
        def style_role_morale(row):
            if row["Avg Morale"] >= 4.0: return ["background-color:#d4edda"] * len(row)
            if row["Avg Morale"] < 3.5:  return ["background-color:#fff3cd"] * len(row)
            return [""] * len(row)
        st.dataframe(
            role_morale.style.apply(style_role_morale, axis=1),
            width="stretch", hide_index=True,
        )
    with rcol2:
        st.bar_chart(role_morale.set_index("Role")[["Avg Morale"]], height=220)

    st.divider()

    # ── Individual morale heatmap (all employees, all weeks) ─────────────────
    st.markdown("**Employee × Week Morale Heatmap**")
    st.caption("Each cell = that employee's morale_index for that week. Darker = higher morale. Filtered to show all 52 weeks.")

    pivot = survey_df.pivot_table(
        index="name", columns="week_number", values="morale_index"
    ).round(2)

    def style_morale_cell(val):
        if pd.isna(val): return ""
        if val < 3.0:    return "background-color:#f8d7da;color:#721c24;font-weight:700"
        if val < 3.5:    return "background-color:#fff3cd;color:#856404"
        if val < 4.0:    return "background-color:#d4edda;color:#155724"
        return "background-color:#2d6a4f;color:#ffffff"

    st.dataframe(
        pivot.style.map(style_morale_cell),
        width="stretch",
        height=420,
    )
    st.caption("🟥 < 3.0 (at-risk)  ·  🟨 3.0–3.5 (watch)  ·  🟩 3.5–4.0 (healthy)  ·  🟢 4.0+ (strong)")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EMPLOYEE MORALE PROFILE
# ═════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.subheader("Employee Morale Profile — 52-Week View")

    emp_list       = sorted(emp_df["name"].tolist())
    default_idx    = emp_list.index("Emma Leblanc") if "Emma Leblanc" in emp_list else 0
    sel_name       = st.selectbox("Select employee", emp_list, index=default_idx, key="morale_profile")
    sel_row        = emp_df[emp_df["name"] == sel_name].iloc[0]
    sel_id         = sel_row["emp_id"]
    emp_survey     = survey_df[survey_df["employee_id"] == sel_id].sort_values("week_number")

    if emp_survey.empty:
        st.info("No survey data found for this employee.")
        st.stop()

    # ── Employee info bar ─────────────────────────────────────────────────────
    ci1, ci2, ci3, ci4, ci5 = st.columns(5)
    avg_mi    = emp_survey["morale_index"].mean()
    min_mi    = emp_survey["morale_index"].min()
    max_mi    = emp_survey["morale_index"].max()
    at_risk_wks = int(emp_survey["at_risk_flag"].sum())
    avg_ret   = emp_survey["retention_intent"].mean()

    ci1.metric("Avg Morale Index",    f"{avg_mi:.2f} / 5.0")
    ci2.metric("Lowest Week",         f"{min_mi:.2f}")
    ci3.metric("Highest Week",        f"{max_mi:.2f}")
    ci4.metric("At-Risk Weeks",       at_risk_wks,
               delta="flagged weeks" if at_risk_wks > 0 else "none flagged",
               delta_color="inverse" if at_risk_wks > 0 else "normal")
    ci5.metric("Avg Retention Intent", f"{avg_ret:.1f} / 5.0")

    status_label = (
        "🚨 At-Risk — T4 triggered" if at_risk_wks >= 3
        else "⚠️ Watch — dips present" if min_mi < 3.5
        else "✅ Healthy"
    )
    if at_risk_wks >= 3:
        st.error(f"**{sel_name}** — {status_label}  ·  {at_risk_wks} weeks below 3.0 threshold")
    elif min_mi < 3.5:
        st.warning(f"**{sel_name}** — {status_label}")
    else:
        st.success(f"**{sel_name}** — {status_label}")

    st.divider()

    # ── 52-week morale trend ──────────────────────────────────────────────────
    st.markdown(f"**{sel_name} — Weekly Morale Index (all 52 weeks)**")
    st.caption("Red cells = at-risk weeks (morale_index < 3.0 for 3+ consecutive weeks)")

    trend_df = emp_survey[["week_number", "morale_index", "retention_intent"]].copy()
    trend_df["At-Risk Threshold"] = 3.0
    trend_df = trend_df.rename(columns={
        "week_number":      "Week",
        "morale_index":     "Morale Index",
        "retention_intent": "Retention Intent",
    })
    st.line_chart(trend_df.set_index("Week"), height=260)

    # Highlight at-risk weeks if any
    if at_risk_wks > 0:
        risk_weeks = emp_survey[emp_survey["at_risk_flag"] == 1]["week_number"].tolist()
        st.caption(f"⚠️ At-risk weeks (morale < 3.0 for 3+): **{', '.join(f'Wk {w}' for w in risk_weeks)}**")

    st.divider()

    # ── Factor breakdown for this employee ────────────────────────────────────
    st.markdown("**Morale Factor Scores — Annual Average vs Team Average**")

    emp_factors  = emp_survey[MORALE_FACTORS].mean().round(2)
    team_factors = survey_df[MORALE_FACTORS].mean().round(2)

    factor_compare = pd.DataFrame({
        "Factor":       [MORALE_LABELS[f] for f in MORALE_FACTORS],
        sel_name:       emp_factors.values,
        "Team Average": team_factors.values,
    })

    def style_compare(row):
        idx = list(factor_compare["Factor"]).index(row["Factor"])
        emp_val  = row[sel_name]
        team_val = row["Team Average"]
        diff = emp_val - team_val
        styles = [""] * len(row)
        emp_col_idx = factor_compare.columns.get_loc(sel_name)
        if diff < -0.3:   styles[emp_col_idx] = "background-color:#f8d7da;font-weight:700"
        elif diff < 0:    styles[emp_col_idx] = "background-color:#fff3cd"
        elif diff > 0.1:  styles[emp_col_idx] = "background-color:#d4edda"
        return styles

    col_fc, col_fb = st.columns([2, 3])
    with col_fc:
        st.dataframe(
            factor_compare.style.apply(style_compare, axis=1),
            width="stretch", hide_index=True,
        )
    with col_fb:
        st.bar_chart(factor_compare.set_index("Factor")[[sel_name, "Team Average"]], height=260)

    st.divider()

    # ── Weekly detail table ───────────────────────────────────────────────────
    with st.expander("📋 Full 52-Week Survey Data"):
        detail_cols = [
            "week_number", "week_start_date", "hours_scheduled",
            *MORALE_FACTORS,
            "overall_morale", "retention_intent", "morale_index",
            "consecutive_low_weeks", "at_risk_flag"
        ]
        detail_display = emp_survey[detail_cols].copy()
        detail_display.columns = [
            "Week", "Week Start", "Hrs Sched",
            "Hours", "Comms", "Mgr Floor", "Product", "Pers Time", "Plan",
            "Overall", "Retention", "Morale Idx",
            "Consec Low", "At-Risk"
        ]

        def style_detail(row):
            if row["At-Risk"] == 1:
                return ["background-color:#f8d7da"] * len(row)
            if row["Morale Idx"] < 3.5:
                return ["background-color:#fff3cd"] * len(row)
            return [""] * len(row)

        st.dataframe(
            detail_display.reset_index(drop=True).style.apply(style_detail, axis=1),
            width="stretch", hide_index=True, height=420,
        )
